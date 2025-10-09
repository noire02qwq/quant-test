from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@dataclass
class TransformerConfig:
    feature_dim: int
    num_classes: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    pooling: str = "mean"
    norm_first: bool = True
    embedding_dropout: float = 0.05


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder for window-based classification."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.feature_dim, cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.embedding_dropout)
        self.pos = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=cfg.norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.pooling = cfg.pooling.lower()
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.embed_drop(z)
        z = self.pos(z)
        z = self.encoder(z)
        z = self.norm(z)
        if self.pooling == "last":
            feat = z[:, -1, :]
        else:
            feat = z.mean(dim=1)
        logits = self.head(feat)
        return logits

    def describe(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture": "TimeSeriesTransformer",
            "d_model": self.cfg.d_model,
            "nhead": self.cfg.nhead,
            "num_layers": self.cfg.num_layers,
            "dim_feedforward": self.cfg.dim_feedforward,
            "dropout": self.cfg.dropout,
            "pooling": self.cfg.pooling,
            "num_classes": self.cfg.num_classes,
            "feature_dim": self.cfg.feature_dim,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


__all__ = ["TimeSeriesTransformer", "TransformerConfig"]
