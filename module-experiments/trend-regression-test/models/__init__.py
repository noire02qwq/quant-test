from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .dlinear import DLinearRegressor
from .gru import GRURegressor
from .transformer import TransformerRegressor
from .patchtst import PatchTSTRegressor
from .timesnet import TimesNetRegressor


MODEL_REGISTRY = {
    "transformer": TransformerRegressor,
    "gru": GRURegressor,
    "dlinear": DLinearRegressor,
    "patchtst": PatchTSTRegressor,
    "timesnet": TimesNetRegressor,
}


def create_model(name: str, input_dim: int, seq_len: int, **kwargs: Any) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"未注册的模型：{name}")
    ModelCls = MODEL_REGISTRY[key]
    kwargs.setdefault("seq_len", seq_len)
    kwargs.setdefault("input_dim", input_dim)
    return ModelCls(**kwargs)


__all__ = ["create_model", "MODEL_REGISTRY"]
