from __future__ import annotations

import torch
from torch import nn


class DLinearRegressor(nn.Module):
    """简化版 DLinear，用于趋势回归。"""

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
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend_init = x.mean(dim=1, keepdim=True)
        seasonal_init = x - trend_init
        if self.individual:
            trend_out = []
            seasonal_out = []
            for i in range(self.input_dim):
                t_in = trend_init[:, :, i]
                s_in = seasonal_init[:, :, i]
                trend_out.append(self.trend[i](t_in))
                seasonal_out.append(self.seasonal[i](s_in))
            trend = torch.stack(trend_out, dim=-1).mean(dim=-1)
            seasonal = torch.stack(seasonal_out, dim=-1).mean(dim=-1)
        else:
            trend = self.trend(trend_init.mean(dim=-1))
            seasonal = self.seasonal(seasonal_init.mean(dim=-1))
        combined = torch.cat([trend, seasonal], dim=-1)
        return self.head(combined.squeeze(1))


__all__ = ["DLinearRegressor"]
