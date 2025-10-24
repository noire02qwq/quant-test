from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXPERIMENT_ROOT / "data"
RAW_FILE = DATA_ROOT / "raw" / "TSM.csv"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"

WINDOW_DAYS = 20
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
    """补充标签计算所需指标。"""

    out = df.copy()
    for period in (5, 20, 60):
        out[f"ema_{period}"] = _ema(out["close"], period)
    tr = _true_range(out)
    out["atr"] = tr.rolling(window=14, min_periods=14).mean()
    return out


def _nanmin(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else np.nan


def _nanmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else np.nan


def compute_regression_labels(df: pd.DataFrame) -> pd.Series:
    """依据独立实验 1010 的定义计算回归标签。"""

    labels = pd.Series(np.nan, index=df.index, dtype=float)
    total = len(df)

    for i in range(total):
        row = df.iloc[i]
        atr = row["atr"]
        entry_price = row["close"]
        if pd.isna(atr) or atr <= 0 or pd.isna(entry_price):
            continue

        future_idx = list(range(i + 1, min(i + 1 + WINDOW_DAYS, total)))
        if len(future_idx) < WINDOW_DAYS:
            continue

        future_df = df.iloc[future_idx]
        if future_df[["high", "low", "close"]].isna().any(axis=None):
            continue

        stop_loss = entry_price - STOP_LOSS_MULT * atr
        stop_gain = entry_price + STOP_GAIN_MULT * atr
        stop_diff = STOP_LOSS_MULT * atr

        trigger = None
        for _, future_row in future_df.iterrows():
            if future_row["low"] <= stop_loss:
                trigger = "loss"
                break
            if future_row["high"] >= stop_gain:
                trigger = "gain"
                break

        min_low = future_df["low"].min()
        max_high = future_df["high"].max()
        end_close = future_df.iloc[-1]["close"]

        if trigger == "loss":
            worst_loss = _nanmin(np.array([stop_loss - entry_price, min_low - entry_price], dtype=float))
            y_origin = worst_loss
        elif trigger == "gain":
            best_gain = _nanmax(np.array([stop_gain - entry_price, max_high - entry_price], dtype=float))
            y_origin = best_gain
        else:
            y_origin = end_close - entry_price

        label_value = y_origin / stop_diff if stop_diff != 0 else np.nan
        labels.iloc[i] = label_value

    return labels


def plot_yearly_overview(df: pd.DataFrame, year: int = 2024) -> Path:
    """输出指定年份的 K 线与标签折线图。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    plot_df = df.loc[year_start:year_end].dropna(subset=["ema_5", "ema_20", "ema_60", "label_y"])
    if plot_df.empty:
        raise ValueError(f"{year} 年数据为空或缺少有效标签，请检查数据准备流程。")

    positions = np.arange(len(plot_df))
    fig, (ax_price, ax_label) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]})

    candle_width = 0.6
    for idx, (_, row) in zip(positions, plot_df.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
        ax_price.add_patch(candle)

    ax_price.plot(positions, plot_df["ema_5"], label="EMA5", color="#1f77b4")
    ax_price.plot(positions, plot_df["ema_20"], label="EMA20", color="#ff7f0e")
    ax_price.plot(positions, plot_df["ema_60"], label="EMA60", color="#2ca02c")
    ax_price.set_ylabel("Price")
    ax_price.set_title(f"TSM {year} Daily K-line with EMA")
    ax_price.grid(True, linestyle="--", alpha=0.3)
    ax_price.legend(loc="upper left")

    ax_label.plot(positions, plot_df["label_y"], label="Normalized Label Y", color="#d62728")
    ax_label.axhline(0, color="#555555", linestyle="--", linewidth=1)
    ax_label.set_ylabel("Label Y")
    ax_label.set_xlabel("Trade Date")
    ax_label.grid(True, linestyle="--", alpha=0.3)
    ax_label.legend(loc="upper left")

    tick_step = max(len(plot_df) // 12, 1)
    tick_positions = list(range(0, len(plot_df), tick_step))
    if tick_positions[-1] != len(plot_df) - 1:
        tick_positions.append(len(plot_df) - 1)
    tick_labels = [plot_df.index[min(pos, len(plot_df) - 1)].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_label.set_xticks(tick_positions)
    ax_label.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()
    output_path = OUTPUT_DIR / f"tsm_{year}_label_overview.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_dataset(df: pd.DataFrame) -> Path:
    """保存计算好的标签数据集。"""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "tsm_regression_labels.csv"
    df.to_csv(output_path, index_label="date")
    return output_path


def main() -> None:
    """加载原始 CSV、计算指标与回归标签并输出可视化。"""

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"未找到原始数据文件：{RAW_FILE}，请先运行 prepare_data.py。")

    df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    enriched = compute_indicators(df)
    enriched["label_y"] = compute_regression_labels(enriched)

    dataset_path = save_dataset(enriched)
    plot_path = plot_yearly_overview(enriched, year=2024)
    valid_ratio = enriched["label_y"].notna().mean()
    print(f"标签数据已生成：{dataset_path}")
    print(f"可视化输出：{plot_path}")
    print(f"有效样本占比：{valid_ratio:.2%}")


if __name__ == "__main__":
    main()
