#!/usr/bin/env python3
import torch
import json
import os
from torch.utils.data import Dataset


piece_to_int = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    '.': 0
}

def fen_to_sequence(fen):
    parts = fen.split()
    fen_board = parts[0]

    sequence = []

    rows = fen_board.split('/')
    for row in rows:
        for char in row:
            if char.isdigit():
                sequence.extend([piece_to_int['.']] * int(char))
            else:
                sequence.append(piece_to_int[char])

    return sequence


def get_data(file_path):

    fens = []
    evals = []
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            try:
                lines = file.readlines()
                if lines:
                    for line in lines:
                        line_data = json.loads(line)
                        fen = line_data.get('fen')
                        evaluation = line_data.get('evals', [{}])[0].get('pvs', [{}])[0].get('cp')
                        if fen and evaluation is not None:
                            print(f'{fen} \t {evaluation}')
                            evals.append(evaluation)
                            fens.append(fen)
                else:
                    print(f"The JSON file {file_path} is empty.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")
    return evals, fens

class ChessDataset(Dataset):
    def __init__(self, evals, fens):
        self.evals = evals
        self.fens = fens

    def __len__(self):
        return len(self.evals)

    def __getitem__(self, idx):
        sequence = fen_to_sequence(self.fens[idx])
        evaluation = self.evals[idx]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(evaluation, dtype=torch.float32)
