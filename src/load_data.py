#!/usr/bin/env python3
import torch
import os
import orjson
import numpy as np
from torch.utils.data import Dataset
import mmap


piece_to_int = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    '.': 0
}

def fen_to_sequence(fen):
    parts = fen.split()
    fen_board = parts[0]
    turn = parts[1]

    sequence = np.zeros(65, dtype=np.int8)

    index = 0
    for char in fen_board:
        if char != '/':
            if char.isdigit():
                index += int(char)
            else:
                sequence[index] = piece_to_int[char]
                index += 1

    sequence[64] = 1 if turn == 'w' else 0

    return sequence


def get_data(file_path):

    fens = []
    evals = []
    if os.path.isfile(file_path):
        with open(file_path, 'r+b') as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            for line in iter(mmapped_file.readline, b""):
                try:
                    line_data = orjson.loads(line)
                    fen = line_data.get('fen')
                    evaluation = line_data.get('evals', [{}])[0].get('pvs', [{}])[0].get('cp')
                    if fen and evaluation is not None:
                        evals.append(evaluation)
                        fens.append(fen)
                except orjson.JSONDecodeError as e:
                    print(f'Error decoding JSON in {file_path}: {e}')
            mmapped_file.close()
    return evals, fens

class ChessDataset(Dataset):
    def __init__(self, evals, fens):
        self.evals = evals
        self.fens = fens
        self.sequences = np.array([fen_to_sequence(fen) for fen in self.fens], dtype=np.int8)

    def __len__(self):
        return len(self.evals)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        evaluation = self.evals[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(evaluation, dtype=torch.float32)
