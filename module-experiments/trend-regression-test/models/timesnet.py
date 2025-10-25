from __future__ import annotations

import math

import torch
from torch import nn


class TimesBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        y = self.conv1(x)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        return x + y


class TimesNetRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        blocks = []
        for i in range(num_blocks):
            kernel = kernel_sizes[i % len(kernel_sizes)]
            blocks.append(TimesBlock(input_dim, hidden_dim, kernel, dropout))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(input_dim)
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        y = x.permute(0, 2, 1)  # (B, C, L)
        for block in self.blocks:
            y = block(y)
        y = y.mean(dim=2)  # (B, C)
        y = self.norm(y)
        return self.head(y)


__all__ = ["TimesNetRegressor"]
