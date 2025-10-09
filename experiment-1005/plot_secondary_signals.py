from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from indicator_pipeline import (
    compute_secondary_signals,
    compute_traditional_indicators,
    load_symbol_frame,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SYMBOL = "AAPL"
YEAR = 2024
EMA_COLUMNS = ("ema_20", "ema_60")


SignalStyle = Tuple[str, str]


SIGNAL_STYLES: Dict[str, SignalStyle] = {
    "signal_macd": ("MACD Golden Cross", "#f94144"),
    "signal_kdj": ("KDJ Golden Cross", "#577590"),
    "signal_ema": ("EMA Bullish Break", "#f3722c"),
    "signal_sar": ("SAR Flip", "#43aa8b"),
    "signal_dmi": ("DMI Cross", "#90be6d"),
    "signal_adtm": ("ADTM Break", "#f9c74f"),
    "signal_ddi": ("DDI Positive", "#277da1"),
    "signal_dpo": ("DPO Positive", "#7209b7"),
    "signal_osc": ("OSC Cross", "#ff9f1c"),
    "signal_srmi": ("SRMI Positive", "#2ec4b6"),
}


def _prepare_subset(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    subset = df.loc[start:end].copy()
    if subset.empty:
        raise ValueError("Selected subset is empty; check symbol/year")
    return subset


def _find_segments(signal: pd.Series) -> List[Tuple[int, int]]:
    indices: List[Tuple[int, int]] = []
    start_idx: int | None = None
    for idx, flag in enumerate(signal.values):
        if flag and start_idx is None:
            start_idx = idx
        elif not flag and start_idx is not None:
            indices.append((start_idx, idx - 1))
            start_idx = None
    if start_idx is not None:
        indices.append((start_idx, len(signal) - 1))
    return indices


def plot_secondary_signals(symbol: str, year: int) -> Path:
    raw = load_symbol_frame(symbol)
    features = compute_traditional_indicators(raw)
    enriched = compute_secondary_signals(features)
    subset = _prepare_subset(enriched, year)

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

    for ema_col in EMA_COLUMNS:
        if ema_col in subset.columns:
            ax_price.plot(positions, subset[ema_col], label=ema_col.upper(), linewidth=1.4)

    legend_handles = list(ax_price.get_legend_handles_labels()[0])
    legend_labels = list(ax_price.get_legend_handles_labels()[1])

    for signal_name, (label, color) in SIGNAL_STYLES.items():
        if signal_name not in subset.columns:
            continue
        segments = _find_segments(subset[signal_name].fillna(False))
        if not segments:
            continue
        patch = plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.18)
        patch.set_label(label)
        legend_handles.append(patch)
        legend_labels.append(label)
        for start, end in segments:
            ax_price.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.18, zorder=0)

    step = max(len(subset) // 10, 1)
    tick_positions = list(range(0, len(subset), step))
    if tick_positions[-1] != len(subset) - 1:
        tick_positions.append(len(subset) - 1)
    tick_labels = [subset.index[pos].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_price.set_xticks(tick_positions)
    ax_price.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax_price.set_title(f"{symbol} {year} Daily K-line with Secondary Signals")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)
    ax_price.legend(legend_handles, legend_labels, loc="upper left", frameon=False)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"secondary_signals_{symbol.lower()}_{year}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(plot_secondary_signals(SYMBOL, YEAR))
