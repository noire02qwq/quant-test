"""根据交易纪律生成中长期/短期标签并可视化胜负情况。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_US

# 复用现有的指标与数据工具
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from technical_indicators import (  # type: ignore
    DATA_DIR,
    IndicatorParams,
    TechnicalIndicatorCalculator,
    fetch_price_data,
)


def _calculate_labels(
    df: pd.DataFrame,
    window_days: int,
    stop_loss_mult: float,
    stop_gain_mult: float,
) -> pd.Series:
    """按照止盈止损规则计算标签（胜/平/负=1/0/-1），窗口按自然日计。"""

    labels = pd.Series(np.nan, index=df.index, dtype=float)
    dates = df.index.to_list()
    total = len(df)

    for i in range(total):
        entry_row = df.iloc[i]
        if pd.isna(entry_row[["open", "atr"]]).any():
            continue

        entry_date = dates[i]
        horizon_end = entry_date + pd.Timedelta(days=window_days)
        future_end = df.index.searchsorted(horizon_end, side="right") - 1
        if future_end <= i:
            continue

        stop_loss = entry_row["open"] - stop_loss_mult * entry_row["atr"]
        stop_gain = entry_row["open"] + stop_gain_mult * entry_row["atr"]
        outcome = 0.0

        for j in range(i, min(future_end + 1, total)):
            future_row = df.iloc[j]
            if pd.isna(future_row[["high", "low"]]).any():
                continue
            if future_row["low"] <= stop_loss:
                outcome = -1.0
                break
            if future_row["high"] >= stop_gain:
                outcome = 1.0
                break

        labels.iloc[i] = outcome

    return labels


def _find_segments(signal: pd.Series) -> List[Tuple[int, int]]:
    """将布尔序列拆分为若干连续区间。"""

    segments: List[Tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(signal.values):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(signal) - 1))
    return segments


def _plot_with_labels(
    df: pd.DataFrame,
    long_labels: pd.Series,
    short_labels: pd.Series,
    long_title: str,
    short_title: str,
) -> None:
    """绘制长短线 K 线图（上中）与 ATR 子图（下）。"""

    df_plot = df.copy()
    positions = np.arange(len(df_plot))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1], hspace=0.3)

    ax_long = fig.add_subplot(gs[0, 0])
    ax_short = fig.add_subplot(gs[1, 0], sharex=ax_long)
    ax_atr = fig.add_subplot(gs[2, 0], sharex=ax_long)

    def _draw_candles(ax):
        candle_width = 0.6
        for idx, (_, row) in zip(positions, df_plot.iterrows()):
            color = "red" if row["close"] >= row["open"] else "green"
            ax.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
            lower = min(row["open"], row["close"])
            height = max(row["open"], row["close"]) - lower or 1e-10
            candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
            ax.add_patch(candle)

    _draw_candles(ax_long)
    stop_loss_long = df_plot["open"] - 2 * df_plot["atr"]
    stop_gain_long = df_plot["open"] + 3 * df_plot["atr"]
    ax_long.plot(positions, stop_loss_long, color="#d62728", linestyle="--", label="Stop Loss (Open - 2 ATR)")
    ax_long.plot(positions, stop_gain_long, color="#2ca02c", linestyle="--", label="Take Profit (Open + 3 ATR)")

    _draw_candles(ax_short)
    stop_loss_short = df_plot["open"] - 1 * df_plot["atr"]
    stop_gain_short = df_plot["open"] + 1.5 * df_plot["atr"]
    ax_short.plot(positions, stop_loss_short, color="#d62728", linestyle="--", label="Stop Loss (Open - 1 ATR)")
    ax_short.plot(positions, stop_gain_short, color="#2ca02c", linestyle="--", label="Take Profit (Open + 1.5 ATR)")

    def _highlight(ax, label_series):
        win_segments = _find_segments(label_series == 1)
        loss_segments = _find_segments(label_series == -1)
        for start, end in win_segments:
            ax.axvspan(start - 0.5, end + 0.5, color="#ffaaaa", alpha=0.3, zorder=0)
        for start, end in loss_segments:
            ax.axvspan(start - 0.5, end + 0.5, color="#90ee90", alpha=0.3, zorder=0)

    _highlight(ax_long, long_labels)
    _highlight(ax_short, short_labels)

    ax_long.set_title(long_title)
    ax_long.set_ylabel("Price")
    ax_long.grid(True, linestyle="--", alpha=0.3)
    ax_long.legend(loc="upper left")

    ax_short.set_title(short_title)
    ax_short.set_ylabel("Price")
    ax_short.grid(True, linestyle="--", alpha=0.3)
    ax_short.legend(loc="upper left")

    ax_atr.plot(positions, df_plot["atr"], color="#1f77b4", label="ATR")
    ax_atr.set_ylabel("ATR")
    ax_atr.grid(True, linestyle="--", alpha=0.3)
    ax_atr.legend(loc="upper left")

    tick_step = max(len(df_plot) // 10, 1)
    tick_positions = list(range(0, len(df_plot), tick_step))
    if tick_positions[-1] != len(df_plot) - 1:
        tick_positions.append(len(df_plot) - 1)
    tick_labels = [df_plot.index[min(pos, len(df_plot) - 1)].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_atr.set_xticks(tick_positions)
    ax_atr.set_xticklabels(tick_labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """计算 TSM 标签并绘制 K 线图与 ATR。"""

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"未找到 qlib 数据目录：{DATA_DIR}，请先运行 build_us_dataset.py 构建数据。"
        )

    qlib.init(provider_uri=str(DATA_DIR), region=REG_US)

    symbol = "TSM"
    plot_start = pd.Timestamp("2024-09-15")
    plot_end = pd.Timestamp("2025-09-15")

    max_window = 15
    start_buffer = 60
    forward_buffer = 30

    fetch_start = (plot_start - pd.Timedelta(days=start_buffer)).strftime("%Y-%m-%d")
    fetch_end = (plot_end + pd.Timedelta(days=forward_buffer)).strftime("%Y-%m-%d")

    price_df = fetch_price_data(symbol, fetch_start, fetch_end)
    calculator = TechnicalIndicatorCalculator(price_df, IndicatorParams())
    indicator_df = calculator.compute()

    long_labels = _calculate_labels(indicator_df, window_days=15, stop_loss_mult=2.0, stop_gain_mult=3.0)
    short_labels = _calculate_labels(indicator_df, window_days=5, stop_loss_mult=1.0, stop_gain_mult=1.5)

    combined = indicator_df.assign(long_label=long_labels, short_label=short_labels)
    combined = combined.loc[: (plot_end + pd.Timedelta(days=0))]

    valid_index = combined.dropna(subset=["long_label", "short_label"]).index
    if len(valid_index) == 0 or valid_index.max() < plot_end:
        raise ValueError("指定区间的末尾缺少足够的数据用于标签计算，请延长数据区间。")

    plot_df = combined.loc[plot_start:plot_end].copy()
    _plot_with_labels(
        plot_df,
        plot_df["long_label"],
        plot_df["short_label"],
        "Long-term Strategy (Stop Loss: -2 ATR, Take Profit: +3 ATR)",
        "Short-term Strategy (Stop Loss: -1 ATR, Take Profit: +1.5 ATR)",
    )

    win_long = int((plot_df["long_label"] == 1).sum())
    loss_long = int((plot_df["long_label"] == -1).sum())
    neutral_long = int((plot_df["long_label"] == 0).sum())

    win_short = int((plot_df["short_label"] == 1).sum())
    loss_short = int((plot_df["short_label"] == -1).sum())
    neutral_short = int((plot_df["short_label"] == 0).sum())

    summary = (
        f"Long-term labels: win={win_long}, loss={loss_long}, neutral={neutral_long}\n"
        f"Short-term labels: win={win_short}, loss={loss_short}, neutral={neutral_short}"
    )
    print(summary)


if __name__ == "__main__":
    main()
