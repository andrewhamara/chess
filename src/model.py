#!/usr/bin/env python3
import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import os
import time

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from load_data import get_data, ChessDataset, piece_to_int

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=65):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].to(x.device)

class ChessTransformer(nn.Module):
    def __init__(self, embedding_dim=256, nhead=8, num_encoder_layers=6, dim_feedforward=512, max_len=65):
        super(ChessTransformer, self).__init__()
        self.embedding = nn.Embedding(len(piece_to_int), embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = torch.bmm(anchor.unsqueeze(1), positive.transpose(1, 2)) / self.temperature
        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)) / self.temperature

        pos_sim = torch.exp(pos_sim).sum(dim=2)
        neg_sim = torch.exp(neg_sim).sum(dim=2)

        denominator = pos_sim + neg_sim

        loss = -torch.log(pos_sim / denominator)
        return loss.mean()

############################ Distributed setup and cleanup #####################

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Process {rank} set up with world size {world_size} and device {torch.cuda.current_device()}", flush=True)

def cleanup():
    dist.destroy_process_group()

#################################################################################

def find_pos_neg_samples(sequences, evaluations, target, num_contrasted_samples):
    pos, neg = [], []

    for idx, ev in enumerate(evaluations):
        if abs(ev - target) < 100:
            pos.append(idx)
        else:
            neg.append(idx)

        if len(pos) >= num_contrasted_samples // 2 and len(neg) >= num_contrasted_samples // 2:
            break

    num_pos_neg = min(len(pos), len(neg), num_contrasted_samples // 2)
    sampled_pos = random.sample(pos, num_pos_neg) if num_pos_neg > 0 else []
    sampled_neg = random.sample(neg, num_pos_neg) if num_pos_neg > 0 else []

    pos_sequences = [sequences[idx] for idx in sampled_pos]
    neg_sequences = [sequences[idx] for idx in sampled_neg]

    return pos_sequences, neg_sequences

def train(rank, world_size, file_path, num_epochs=10, num_contrasted_samples=100):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    print(f'Process {rank} using device {device}', flush=True)

    evals, fens = get_data(file_path)
    dataset = ChessDataset(evals, fens)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=36)

    model = ChessTransformer().to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    criterion = InfoNCELoss(temperature=0.07).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    dataset_size = len(dataset)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_index, (sequences, evaluations) in enumerate(dataloader):
            sequences, evaluations = sequences.to(device), evaluations.to(device)
            optimizer.zero_grad()

            anchor_embeddings = model(sequences)
            pos_sequences, neg_sequences = [], []
            batch_size = sequences.size(0)

            for cur_chunk in range(0, dataset_size, 10000):
                chunk_end = min(cur_chunk + 10000, dataset_size)
                chunk_sequences = dataset.sequences[cur_chunk:chunk_end]
                chunk_evals = dataset.evals[cur_chunk:chunk_end]

                for i in range(len(evaluations)):
                    target = evaluations[i].item()
                    pos, neg = find_pos_neg_samples(chunk_sequences, chunk_evals, target, num_contrasted_samples)
                    pos_sequences.extend(pos)
                    neg_sequences.extend(neg)

                if len(pos_sequences) >= batch_size * (num_contrasted_samples // 2) and \
                   len(neg_sequences) >= batch_size * (num_contrasted_samples // 2):
                    break

            pos_sequences = pos_sequences[:batch_size * (num_contrasted_samples // 2)]
            neg_sequences = neg_sequences[:batch_size * (num_contrasted_samples // 2)]

            pos_emb = model(torch.stack([torch.tensor(seq, dtype=torch.long).to(device) for seq in pos_sequences]))
            neg_emb = model(torch.stack([torch.tensor(seq, dtype=torch.long).to(device) for seq in neg_sequences]))

            positives = pos_emb.view(batch_size, num_contrasted_samples // 2, -1)
            negatives = neg_emb.view(batch_size, num_contrasted_samples // 2, -1)
            loss = criterion(anchor_embeddings, positives, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f'Batch [{batch_index+1}/{len(dataloader)}] completed by process {rank}', flush=True)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f} by process {rank}', flush=True)
    cleanup()
    if rank == 0:
        torch.save(model.state_dict(), '/data/hamaraa/model_small.pth')

if __name__ == '__main__':
    file_path = '/data/hamaraa/data.json'
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}", flush=True)
    mp.spawn(train,
             args=(world_size, file_path),
             nprocs=world_size,
             join=True)
