from __future__ import annotations

import torch
from torch import nn


class DLinearRegressor(nn.Module):
    """简化版 DLinear 模型：趋势/季节分解 + 线性回归。"""

    def __init__(self, input_dim: int, seq_len: int, individual: bool = False) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.individual = individual
        if individual:
            self.trend = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(input_dim)])
            self.seasonal = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(input_dim)])
        else:
            self.trend = nn.Linear(seq_len, 1)
            self.seasonal = nn.Linear(seq_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        trend_init = x.mean(dim=1, keepdim=True)
        seasonal_init = x - trend_init
        if self.individual:
            trend_out = []
            seasonal_out = []
            for i in range(self.input_dim):
                inp_t = trend_init[:, :, i]
                inp_s = seasonal_init[:, :, i]
                trend_out.append(self.trend[i](inp_t))
                seasonal_out.append(self.seasonal[i](inp_s))
            trend = torch.stack(trend_out, dim=-1).mean(dim=-1)
            seasonal = torch.stack(seasonal_out, dim=-1).mean(dim=-1)
        else:
            trend = self.trend(trend_init.mean(dim=-1))
            seasonal = self.seasonal(seasonal_init.mean(dim=-1))
        return trend + seasonal


__all__ = ["DLinearRegressor"]
