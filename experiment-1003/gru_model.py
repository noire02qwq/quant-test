"""基于 GRU 的时间序列分类模型（简化版本）。

说明：
- 仅依赖 torch.nn 中的标准模块：GRU + LayerNorm + 线性分类头。
- 输入为 [B, T, F]（批次、时间窗长度、特征维度）。
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """GRU 分类模型：GRU -> 归一化 -> 全连接分类。

    参数：
    - feature_dim: 输入特征维度 F
    - hidden_size: GRU 隐层维度
    - num_layers: GRU 堆叠层数
    - dropout: GRU 层间 dropout（当 num_layers > 1 生效）
    - bidirectional: 是否双向 GRU（默认 False）
    - pooling: 池化方式，"last" 取最后时刻，"mean" 取时间维平均
    - num_classes: 分类类别数（胜/平/负=3）
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 4096,
        num_layers: int = 64,
        dropout: float = 0.25,
        bidirectional: bool = False,
        pooling: str = "mean",
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.pooling = pooling
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ELU(), nn.Dropout(dropout), nn.Linear(out_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        z, h_n = self.gru(x)  # z: [B, T, H], h_n: [num_layers*(1/2), B, H]
        if self.pooling == "mean":
            feat = z.mean(dim=1)
        else:
            feat = z[:, -1, :]
        feat = self.norm(feat)
        logits = self.head(feat)
        return logits


def _count_params(m: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    """快速查看模型结构与参数量。"""
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import load_dataframe, prepare_splits  # type: ignore

    cfg = json.loads(Path("experiment-1003/config.json").read_text(encoding="utf-8"))
    df = load_dataframe(Path(cfg["data_path"]), index_col=cfg.get("index_col", "datetime"))
    _, _, _, feature_cols = prepare_splits(df, cfg, window_size=int(cfg["window_size"]))

    model = GRUClassifier(feature_dim=len(feature_cols))
    print({
        "info": "model input spec",
        "per_timestep_vector_dim": len(feature_cols),
        "example_input_shape": [1, int(cfg["window_size"]), len(feature_cols)],
    })
    print(model)
    total, trainable = _count_params(model)
    print({"feature_dim": len(feature_cols), "total_params": total, "trainable_params": trainable})


if __name__ == "__main__":
    main()
