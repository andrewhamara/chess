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
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
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

        pos_sim = torch.exp(pos_sim).squeeze(1)
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

def find_pos_neg_samples(sequences, evaluations, target, num_negatives):
    pos, neg = None, []

    for idx, ev in enumerate(evaluations):
        if pos is None and abs(ev - target) < 100:
            pos = idx
        elif abs(ev - target) >= 100 and len(neg) < num_negatives:
            neg.append(idx)
        
        if pos is not None and len(neg) == num_negatives:
            break

    if pos is None or len(neg) < num_negatives:
        return None, None


    pos_sequence = sequences[pos]
    neg_sequences = [sequences[idx] for idx in neg]

    return pos_sequence, neg_sequences

def train(rank, world_size, file_path, save_path, num_epochs=10, num_negatives=50):
    print('setting up...', flush=True)
    setup(rank, world_size)

    device = torch.device(f'cuda:{rank}')
    print(f'Process {rank} using device {device}', flush=True)

    print('getting data...', flush=True)
    evals, fens = get_data(file_path)

    print('Creating dataset...', flush=True)
    dataset = ChessDataset(evals, fens)

    print('creating sampler...', flush=True)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    print('creating dataloader...', flush=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=train_sampler, num_workers=4)

    print('creating model...', flush=True)
    model = ChessTransformer().to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    print('creating loss and optimizer...', flush=True)
    criterion = InfoNCELoss(temperature=0.07).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('starting to train...', flush=True)
    model.train()
    dataset_size = len(dataset)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_index, (sequences, evaluations) in enumerate(dataloader):
            sequences, evaluations = sequences.to(device), evaluations.to(device)
            optimizer.zero_grad()

            anchor_embeddings = model(sequences)
            pos_embeddings_list, neg_embeddings_list = [], []
            batch_size = sequences.size(0)

            for i in range(batch_size):
                target = evaluations[i].item()
                cur_chunk = 0
                negative_samples = []
                positive_sample = None
                positive_found = False

                while len(negative_samples) < num_negatives:
                    chunk_end = min(cur_chunk + 10000, dataset_size)
                    chunk_sequences = dataset.sequences[cur_chunk:chunk_end]
                    chunk_evals = dataset.evals[cur_chunk:chunk_end]
                    cur_chunk = chunk_end

                    pos_seq, neg_seqs = find_pos_neg_samples(chunk_sequences, chunk_evals, target, num_negatives - len(negative_samples))
                    if pos_seq is not None and not positive_found:
                        positive_sample = pos_seq
                        positive_found = True
                    if neg_seqs is not None:
                        negative_samples.extend(neg_seqs)

                    if cur_chunk >= dataset_size:
                        break

                if positive_found and len(negative_samples) == num_negatives:
                    positive_embedding = model(torch.tensor(positive_sample, dtype=torch.long).unsqueeze(0).to(device))
                    negative_embeddings = model(torch.stack([torch.tensor(seq, dtype=torch.long).to(device) for seq in negative_samples]))

                    pos_embeddings_list.append(positive_embedding)
                    neg_embeddings_list.append(negative_embeddings)

            if len(pos_embeddings_list) == batch_size and len(neg_embeddings_list) == batch_size:

                positive_embeddings = torch.cat(pos_embeddings_list, dim=0)
                negative_embeddings = torch.cat(neg_embeddings_list, dim=0)
                negative_embeddings = negative_embeddings.view(batch_size, num_negatives, -1)

                loss = criterion(anchor_embeddings, positive_embeddings.unsqueeze(1), negative_embeddings)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            print(f'Batch [{batch_index+1}/{len(dataloader)}] completed by process {rank}', flush=True)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f} by process {rank}', flush=True)
        if rank == 0:
            epoch_save_path = os.path.join(save_path, f'chess_small_{epoch+1}.pth')
            torch.save(model.state_dict(), epoch_save_path)
            print(f'Model saved at {epoch_save_path}', flush=True)
    cleanup()

if __name__ == '__main__':
    file_path = '/data/hamaraa/chess_1m.json'
    save_path = '/data/hamaraa/model_checkpoints'
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}", flush=True)
    mp.spawn(train,
             args=(world_size, file_path, save_path),
             nprocs=world_size,
             join=True)
