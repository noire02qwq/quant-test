"""Model registry for preposition-experiments/experiment-1005."""

from __future__ import annotations

from typing import Dict

from .transformer import TimeSeriesTransformer, TransformerConfig

MODEL_REGISTRY: Dict[str, type] = {
    "transformer": TimeSeriesTransformer,
}

__all__ = ["MODEL_REGISTRY", "TimeSeriesTransformer", "TransformerConfig"]
