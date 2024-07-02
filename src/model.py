#!/usr/bin/env python3
import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random

from load_data import get_data, ChessDataset, piece_to_int


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
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
    def __init__(self, embedding_dim=1280, nhead=8, num_encoder_layers=6, dim_feedforward=512, max_len=64):
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

        pos_sim = torch.exp(torch.sum(anchor * positive, dim=-1) / self.temperature)
        neg_sim = torch.exp(torch.mm(anchor, negatives.t()) / self.temperature)

        denominator = pos_sim + torch.sum(neg_sim, dim=-1)

        loss = -torch.log(pos_sim / denominator)
        return loss.mean()


def train(model, dataloader, dataset, criterion, optimizer, num_epochs=10, num_contrasted_samples=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for sequence, evaluation in dataloader:
            optimizer.zero_grad()

            anchor_embedding = model(sequence)

            random_indices = random.sample(range(len(dataset)), num_contrasted_samples)
            random_samples = [dataset[i] for i in random_indices]
            random_sequences = torch.stack([sample[0] for sample in random_samples])
            random_evals = torch.stack([sample[1] for sample in random_samples])

            random_embeddings = model(random_sequences)

            pos_mask = torch.abs(evaluation.unsqueeze(1) - random_evals.unsqueeze(0)) < 100
            neg_mask = torch.abs(evaluation.unsqueeze(1) - random_evals.unsqueeze(0)) >= 100

            if pos_mask.any() and neg_mask.any():
                positive_indices = pos_mask.nonzero(as_tuple=True)[1]
                negative_indices = neg_mask.nonzero(as_tuple=True)[1]

                positive_samples = random_embeddings[positive_indices]
                negative_samples = random_embeddings[negative_indices]

                loss = criterion(anchor_embedding, positive_samples, negative_samples)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print('did a sequence')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

if __name__ == '__main__':
    file_path = 'path_to_json_here'
    evals, fens = get_data(file_path)
    dataset = ChessDataset(evals, fens)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    embedding_dim = 1280
    model = ChessTransformer()
    criterion = InfoNCELoss(temperature=0.07)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, dataset, criterion, optimizer)
