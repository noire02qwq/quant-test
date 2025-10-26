from __future__ import annotations

import math

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim: int, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.proj = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape
        if C != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {C}")
        if L < self.patch_len:
            raise ValueError("sequence length shorter than patch length")
        unfolded = []
        for start in range(0, L - self.patch_len + 1, self.stride):
            end = start + self.patch_len
            patch = x[:, start:end, :]  # (B, patch_len, C)
            unfolded.append(patch.reshape(B, -1))
        patches = torch.stack(unfolded, dim=1)  # (B, num_patches, patch_len*C)
        return self.proj(patches)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        input_dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(input_dim, patch_len, stride, d_model)
        self.input_dropout = nn.Dropout(input_dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling.lower()
        if self.pooling not in {"mean", "last"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        patches = self.patch_embed(x)
        patches = self.input_dropout(patches)
        patches = self.pos_encoder(patches)
        encoded = self.encoder(patches)
        if self.pooling == "mean":
            pooled = encoded.mean(dim=1)
        else:
            pooled = encoded[:, -1, :]
        pooled = self.norm(pooled)
        return self.head(pooled)


__all__ = ["PatchTSTRegressor"]
