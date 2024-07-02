#!/usr/bin/env python3
import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader

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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (seq1, eval1) in enumerate(dataloader):
            for j, (seq2, eval2) in enumerate(dataloader):
                if i != j:
                    label = torch.tensor([1.0 if abs(eval1.item() - eval2.item()) < 100 else 0.0], dtype=torch.float32)
                    optimizer.zero_grad()
                    embedding1 = model(seq1)
                    embedding2 = model(seq2)
                    loss = criterion(embedding1, embedding2, label)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1},{j+1}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss / len(dataloader):.4f}')

if __name__ == '__main__':
    file_path = '/Volumes/andy/splits/chunk_1.json'
    evals, fens = get_data(file_path)
    dataset = ChessDataset(evals, fens)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    embedding_dim = 1280
    model = ChessTransformer(embedding_dim=embedding_dim, max_len=100)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, criterion, optimizer, num_epochs=10)
