from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .dlinear import DLinearClassifier
from .gru import GRUClassifier
from .transformer import TransformerClassifier


MODEL_REGISTRY = {
    "transformer": TransformerClassifier,
    "gru": GRUClassifier,
    "dlinear": DLinearClassifier,
}


def create_model(name: str, input_dim: int, seq_len: int, num_classes: int = 3, **kwargs: Any) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"未注册的模型：{name}")
    ModelCls = MODEL_REGISTRY[key]
    if key == "dlinear":
        kwargs.setdefault("seq_len", seq_len)
        kwargs.setdefault("input_dim", input_dim)
        kwargs.setdefault("num_classes", num_classes)
        return ModelCls(**kwargs)
    return ModelCls(input_dim=input_dim, num_classes=num_classes, **kwargs)


__all__ = ["create_model", "MODEL_REGISTRY"]
