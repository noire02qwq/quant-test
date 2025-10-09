from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from indicator_pipeline import (
    compute_long_strategy_labels,
    compute_traditional_indicators,
    load_symbol_frame,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SYMBOL = "AAPL"
YEAR = 2024
MA_COLUMNS = ("ma_5", "ma_20", "ma_60")
HIGHLIGHT_COLOR = "#ff4d4f"


def _prepare_subset(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    subset = df.loc[start:end].copy()
    if subset.empty:
        raise ValueError(f"{SYMBOL} 在 {year} 年没有可用数据，检查输入 CSV。")
    return subset


def _find_segments(signal: pd.Series) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start_idx: int | None = None
    for idx, flag in enumerate(signal.values):
        if flag and start_idx is None:
            start_idx = idx
        elif not flag and start_idx is not None:
            segments.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(signal) - 1))
    return segments


def plot_long_strategy_labels(symbol: str, year: int) -> Path:
    raw = load_symbol_frame(symbol)
    features = compute_traditional_indicators(raw)
    labels = compute_long_strategy_labels(features)
    features = features.assign(long_win=labels)
    subset = _prepare_subset(features, year)

    if "long_win" not in subset.columns:
        raise ValueError("长线策略标签缺失，无法绘图。")

    positions = np.arange(len(subset))
    fig, ax_price = plt.subplots(figsize=(16, 6))

    candle_width = 0.6
    for idx, (_, row) in enumerate(subset.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax_price.vlines(idx, row["low"], row["high"], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower
        height = height if height > 0 else 1e-4
        ax_price.add_patch(plt.Rectangle((idx - candle_width / 2, lower), candle_width, height, color=color, alpha=0.8))

    for ma in MA_COLUMNS:
        if ma in subset.columns:
            ax_price.plot(positions, subset[ma], label=ma.upper(), linewidth=1.5)

    legend_handles, legend_labels = ax_price.get_legend_handles_labels()

    win_segments = _find_segments(subset["long_win"].fillna(0).astype(bool))
    if win_segments:
        highlight_patch = plt.Rectangle((0, 0), 1, 1, color=HIGHLIGHT_COLOR, alpha=0.18)
        legend_handles.append(highlight_patch)
        legend_labels.append("15-day Long Strategy Win")
        for start, end in win_segments:
            ax_price.axvspan(start - 0.5, end + 0.5, color=HIGHLIGHT_COLOR, alpha=0.18, zorder=0)

    step = max(len(subset) // 10, 1)
    tick_positions = list(range(0, len(subset), step))
    if tick_positions[-1] != len(subset) - 1:
        tick_positions.append(len(subset) - 1)
    tick_labels = [subset.index[pos].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_price.set_xticks(tick_positions)
    ax_price.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax_price.set_title(f"{symbol} {year} Daily K-line with 15-day Long Strategy Labels")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)
    if legend_handles:
        ax_price.legend(legend_handles, legend_labels, loc="upper left", frameon=False)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"long_strategy_labels_{symbol.lower()}_{year}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(plot_long_strategy_labels(SYMBOL, YEAR))
