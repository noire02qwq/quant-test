"""模型主体：基于 Transformer Encoder 的时间序列分类模型。"""

from __future__ import annotations

import math
from dataclasses import dataclass
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """时间序列分类模型：Linear 投影 + PosEncoding + TransformerEncoder + 池化 + 分类头。"""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",
        num_classes: int = 2,
        norm_first: bool = True,
        embedding_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, d_model)
        self.embed_drop = nn.Dropout(embedding_dropout)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.RReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
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


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    """简单主程序：读取配置，构建模型，打印结构与参数量。"""
    import json
    from pathlib import Path
    from data import load_dataframe, prepare_frames

    cfg = json.loads(Path("experiment-1003/config.json").read_text(encoding="utf-8"))
    df = load_dataframe(Path(cfg["data_path"]), index_col=cfg.get("index_col", "datetime"))
    train_df, val_df, test_df, feature_cols = prepare_frames(df, cfg)

    mcfg = cfg["model"]
    model = TimeSeriesTransformer(
        feature_dim=len(feature_cols),
        d_model=int(mcfg.get("d_model", 128)),
        nhead=int(mcfg.get("nhead", 4)),
        num_layers=int(mcfg.get("num_layers", 4)),
        dim_feedforward=int(mcfg.get("dim_feedforward", 256)),
        dropout=float(mcfg.get("dropout", 0.1)),
        pooling=str(mcfg.get("pooling", "mean")),
        num_classes=2,
        norm_first=bool(mcfg.get("norm_first", True)),
        embedding_dropout=float(mcfg.get("embedding_dropout", 0.05)),
    )
    print({
        "info": "model input spec",
        "per_timestep_vector_dim": len(feature_cols),
        "example_input_shape": [1, int(cfg["window_size"]), len(feature_cols)],
    })
    print(model)
    total, trainable = _count_parameters(model)
    print({"feature_dim": len(feature_cols), "total_params": total, "trainable_params": trainable})


if __name__ == "__main__":
    main()
