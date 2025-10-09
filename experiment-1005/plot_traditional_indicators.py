from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from indicator_pipeline import (
    DATA_ROOT,
    compute_traditional_indicators,
    load_symbol_frame,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SYMBOL = "AAPL"
YEAR = 2024
MA_COLUMNS = ("ma_5", "ma_20", "ma_60")


def _prepare_subset(df: pd.DataFrame, year: int) -> pd.DataFrame:
    mask = (df.index >= pd.Timestamp(year=year, month=1, day=1)) & (
        df.index <= pd.Timestamp(year=year, month=12, day=31)
    )
    subset = df.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"{SYMBOL} 在 {year} 年没有可用数据，检查输入 CSV。")
    return subset


def _plot_candles(ax: plt.Axes, df: pd.DataFrame) -> None:
    positions = np.arange(len(df))
    candle_width = 0.6
    for idx, (_, row) in enumerate(df.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax.vlines(idx, row["low"], row["high"], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower
        height = height if height > 0 else 1e-4
        rect = plt.Rectangle((idx - candle_width / 2, lower), candle_width, height, color=color, alpha=0.8)
        ax.add_patch(rect)
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.3)


def _format_xaxis(ax: plt.Axes, df: pd.DataFrame) -> None:
    positions = np.arange(len(df))
    dates = df.index.to_pydatetime()
    ax.set_xticks(positions[:: max(len(df) // 10, 1)])
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates[:: max(len(df) // 10, 1)]], rotation=45, ha="right")


def plot_traditional_indicators(symbol: str, year: int) -> Path:
    raw = load_symbol_frame(symbol)
    features = compute_traditional_indicators(raw)
    subset = _prepare_subset(features, year)

    fig, (ax_price, ax_macd, ax_kdj) = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.2]},
    )

    _plot_candles(ax_price, subset)

    positions = np.arange(len(subset))
    for ma in MA_COLUMNS:
        if ma in subset.columns:
            ax_price.plot(positions, subset[ma], label=ma.upper(), linewidth=1.5)
    ax_price.legend(loc="upper left")
    ax_price.set_title(f"{symbol} {year} Daily K-line with MA")

    macd_hist = subset["macd_hist"]
    ax_macd.bar(positions, macd_hist, color=np.where(macd_hist >= 0, "red", "green"), alpha=0.6, width=0.6)
    ax_macd.plot(positions, subset["macd_dif"], label="MACD DIF", color="#1f77b4")
    ax_macd.plot(positions, subset["macd_dea"], label="MACD DEA", color="#ff7f0e")
    ax_macd.axhline(0.0, color="black", linewidth=0.8)
    ax_macd.legend(loc="upper left")
    ax_macd.set_ylabel("MACD")
    ax_macd.grid(True, linestyle="--", alpha=0.3)

    ax_kdj.plot(positions, subset["kdj_k"], label="K", color="#1f77b4")
    ax_kdj.plot(positions, subset["kdj_d"], label="D", color="#ff7f0e")
    ax_kdj.plot(positions, subset["kdj_j"], label="J", color="#2ca02c")
    ax_kdj.axhline(20, color="#999999", linestyle="--", linewidth=0.8)
    ax_kdj.axhline(80, color="#999999", linestyle="--", linewidth=0.8)
    ax_kdj.set_ylabel("KDJ")
    ax_kdj.legend(loc="upper left")
    ax_kdj.grid(True, linestyle="--", alpha=0.3)

    _format_xaxis(ax_kdj, subset)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"traditional_indicators_{symbol.lower()}_{year}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(plot_traditional_indicators(SYMBOL, YEAR))
