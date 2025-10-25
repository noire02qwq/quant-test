from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXPERIMENT_ROOT / "data"
RAW_FILE = DATA_ROOT / "raw" / "QQQ.csv"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"

LOOKAHEAD_DAYS = 15
STOP_LOSS_MULT = 2.0
STOP_GAIN_MULT = 3.0


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for period in (5, 20, 60):
        out[f"ema_{period}"] = _ema(out["close"], period)
    tr = _true_range(out)
    out["atr"] = tr.rolling(window=14, min_periods=14).mean()
    return out


def compute_binary_labels(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    total = len(df)

    for i in range(total):
        row = df.iloc[i]
        atr = row["atr"]
        entry_price = row["close"]
        if pd.isna(atr) or atr <= 0 or pd.isna(entry_price):
            continue

        future_idx = list(range(i + 1, min(i + 1 + LOOKAHEAD_DAYS, total)))
        if len(future_idx) < LOOKAHEAD_DAYS:
            continue

        future_df = df.iloc[future_idx]
        if future_df[["high", "low"]].isna().any(axis=None):
            continue

        target_high = entry_price + STOP_GAIN_MULT * atr
        target_low = entry_price - STOP_LOSS_MULT * atr

        condition = (future_df["high"].max() > target_high) and (future_df["low"].min() > target_low)
        labels.iloc[i] = 1.0 if condition else 0.0

    return labels


def _plot_candles(ax, df: pd.DataFrame) -> None:
    positions = np.arange(len(df))
    candle_width = 0.6
    for idx, (_, row) in zip(positions, df.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
        ax.add_patch(candle)


def plot_label_overlay(df: pd.DataFrame, year: int = 2024) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    plot_df = df.loc[year_start:year_end].dropna(subset=["ema_5", "ema_20", "ema_60", "label"])
    if plot_df.empty:
        raise ValueError(f"{year} 年数据为空或缺少有效标签，请检查数据准备流程。")

    fig, ax = plt.subplots(figsize=(16, 6))
    _plot_candles(ax, plot_df)
    positions = np.arange(len(plot_df))
    ax.plot(positions, plot_df["ema_5"], label="EMA5", color="#1f77b4")
    ax.plot(positions, plot_df["ema_20"], label="EMA20", color="#ff7f0e")
    ax.plot(positions, plot_df["ema_60"], label="EMA60", color="#2ca02c")

    for idx, label in enumerate(plot_df["label"]):
        color = "green" if label >= 0.5 else "red"
        ax.axvspan(idx - 0.5, idx + 0.5, color=color, alpha=0.08)

    ax.set_title(f"QQQ {year} Daily K-line with Label Overlay")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    tick_step = max(len(plot_df) // 12, 1)
    tick_positions = list(range(0, len(plot_df), tick_step))
    if tick_positions[-1] != len(plot_df) - 1:
        tick_positions.append(len(plot_df) - 1)
    tick_labels = [plot_df.index[min(pos, len(plot_df) - 1)].strftime("%Y-%m-%d") for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()
    output_path = OUTPUT_DIR / f"qqq_{year}_label_overlay.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_dataset(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "qqq_classification_labels.csv"
    df.to_csv(output_path, index_label="date")
    return output_path


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"未找到原始数据文件：{RAW_FILE}，请先运行 prepare_data.py。")

    df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    enriched = compute_indicators(df)
    enriched["label"] = compute_binary_labels(enriched)

    dataset_path = save_dataset(enriched)
    plot_path = plot_label_overlay(enriched, year=2024)
    valid_ratio = enriched["label"].notna().mean()
    positive_ratio = enriched["label"].mean(skipna=True)
    print(f"标签数据已生成：{dataset_path}")
    print(f"可视化输出：{plot_path}")
    print(f"有效样本占比：{valid_ratio:.2%}，正类占比：{positive_ratio:.2%}")


if __name__ == "__main__":
    main()
