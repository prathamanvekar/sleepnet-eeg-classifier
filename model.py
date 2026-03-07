"""
model.py
========
CNN + TCN + Attention for sleep staging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.dropout(F.relu(self.bn(self.conv(x))))
        return F.relu(out + self.residual(x))


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        # x: (batch, seq, dim)
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class SleepNet(nn.Module):
    def __init__(self, n_classes=5, dropout=0.3):
        super().__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 50, stride=6),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, 8, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, 8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )

        # TCN
        self.tcn = nn.Sequential(
            TCNBlock(128, 128, dilation=1, dropout=dropout),
            TCNBlock(128, 128, dilation=2, dropout=dropout),
            TCNBlock(128, 128, dilation=4, dropout=dropout),
        )

        # Attention
        self.attention = Attention(128)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, 1, 3000)
        x = self.cnn(x)           # (batch, 128, ~6)
        x = self.tcn(x)           # (batch, 128, ~6)
        x = x.permute(0, 2, 1)    # (batch, ~6, 128)
        x = self.attention(x)     # (batch, 128)
        x = self.classifier(x)    # (batch, n_classes)
        return x


if __name__ == "__main__":
    model = SleepNet()
    x = torch.randn(4, 1, 3000)
    y = model(x)
    print(f"Input: {x.shape} → Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
