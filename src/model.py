#!/usr/bin/env python3
import torch
#torch.cuda.empty_cache()
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

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

        print('normalized shapes:')
        print(positive.shape)
        print(negatives.shape)
        pos_sim = torch.bmm(anchor.unsqueeze(1), positive.transpose(1, 2)) / self.temperature
        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)) / self.temperature

        pos_sim = torch.exp(pos_sim).sum(dim=2)
        neg_sim = torch.exp(neg_sim).sum(dim=2)

        print('similarity shapes:')
        print(neg_sim.shape)
        print(pos_sim.shape)
        denominator = pos_sim + neg_sim

        loss = -torch.log(pos_sim / denominator)
        return loss.mean()


def train(model, dataloader, dataset, criterion, optimizer, device, num_epochs=10, num_contrasted_samples=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        import time
        for batch_index, (sequences, evaluations) in enumerate(dataloader):
            before = time.time()
            print('starting a batch')
            sequences, evaluations = sequences.to(device), evaluations.to(device)
            optimizer.zero_grad()

            anchor_embeddings = model(sequences)

            pos = []
            neg = []
            batch_size = sequences.size(0)

            for i in range(batch_size):
                pos_indices = [idx for idx, (seq, ev) in enumerate(dataset) if abs(ev - evaluations[i]) < 100]
                neg_indices = [idx for idx, (seq, ev) in enumerate(dataset) if abs(ev - evaluations[i]) >= 100]

                # realistically will always be 100 / 2 = 50, but if dataset is smaller this will help
                num_pos_neg = min(len(pos_indices), len(neg_indices), num_contrasted_samples // 2)
                sampled_positives = random.sample(pos_indices, num_pos_neg)
                sampled_negatives = random.sample(neg_indices, num_pos_neg)

                pos_sequences = [dataset[idx][0] for idx in sampled_positives]
                neg_sequences = [dataset[idx][0] for idx in sampled_negatives]

                pos_emb = model(torch.stack(pos_sequences).to(device))
                neg_emb = model(torch.stack(neg_sequences).to(device))

                pos.append(pos_emb)
                neg.append(neg_emb)

            positives = torch.cat(pos, dim=0)
            negatives = torch.cat(neg, dim=0)
            print(positives.shape)
            print(negatives.shape)

            positives = positives.view(batch_size, 50, -1)
            negatives = negatives.view(batch_size, 50, -1)
            loss = criterion(anchor_embeddings, positives, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            after = time.time()
            elapsed = after - before
            print('took {elapsed} seconds')
            print(f'Batch [{batch_index+1}/{len(dataloader)}]')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    torch.save(model.state_dict(), '/data/hamaraa/model.pth')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    file_path = 'f_name'
    evals, fens = get_data(file_path)
    dataset = ChessDataset(evals, fens)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    embedding_dim = 256
    model = ChessTransformer().to(device)
    criterion = InfoNCELoss(temperature=0.07).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, dataset, criterion, optimizer, device)
