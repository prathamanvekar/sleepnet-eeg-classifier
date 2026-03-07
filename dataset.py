"""
dataset.py
==========
PyTorch Dataset with data augmentation.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SleepDataset(Dataset):

    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 3000)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x):
        # Random amplitude scaling (0.8 - 1.2)
        if np.random.rand() < 0.5:
            scale = 0.8 + 0.4 * np.random.rand()
            x = x * scale

        # Add Gaussian noise
        if np.random.rand() < 0.3:
            noise = 0.05 * torch.randn_like(x)
            x = x + noise

        # Random time shift (up to 10% of signal)
        if np.random.rand() < 0.3:
            shift = int(0.1 * x.shape[-1] * (np.random.rand() - 0.5))
            x = torch.roll(x, shifts=shift, dims=-1)

        return x
