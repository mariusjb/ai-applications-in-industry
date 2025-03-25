import torch
from torch.utils.data import Dataset

class Dataset_ECF(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences
        self.y = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y