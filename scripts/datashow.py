"""Plot TSM 日K与成交量，完全依赖本地构建的 qlib 数据集。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data import D

# 本地 qlib 数据目录（由 build_us_dataset.py 构建）
DATA_DIR = Path(__file__).resolve().parent.parent / "data/qlib_us_selected/qlib_data"


def init_qlib(provider_path: Path) -> None:
    """初始化 qlib，指向本项目生成的数据目录。"""

    qlib.init(provider_uri=str(provider_path), region=REG_US)


def get_instrument_coverage(symbol: str, provider_path: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """读取 instruments/all.txt，获取指定股票的可用时间区间。"""

    instruments_file = provider_path / "instruments" / "all.txt"
    if not instruments_file.exists():
        return None

    with instruments_file.open("r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[0].upper() == symbol.upper():
                return pd.Timestamp(parts[1]), pd.Timestamp(parts[2])
    return None


def load_daily_frame(symbol: str, start: str, end: str) -> pd.DataFrame:
    """通过 qlib 的 D.features 提取 OHLCV，并统一索引格式。"""

    fields = ["$open", "$high", "$low", "$close", "$volume"]
    raw = D.features([symbol], fields=fields, start_time=start, end_time=end, freq="day")
    if raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = raw.xs(symbol, level="instrument")
    df = df.rename(columns={name: name.lstrip("$") for name in df.columns})

    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    else:
        df.index = pd.to_datetime(df.index)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    return df


def plot_candlestick_with_volume(df: pd.DataFrame, symbol: str, start: str, end: str) -> None:
    """使用交易日序号绘制 K 线与成交量，保证横轴连续。"""

    positions = range(len(df))

    fig, (ax_price, ax_volume) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    candle_width = 0.6

    for idx, (_, row) in zip(positions, df.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"

        ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)

        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
        ax_price.add_patch(candle)

        volume_bar = Rectangle((idx - candle_width / 2, 0), candle_width, row["volume"], edgecolor=color, facecolor=color, alpha=0.5)
        ax_volume.add_patch(volume_bar)

    tick_step = max(len(df) // 10, 1)
    tick_positions = list(range(0, len(df), tick_step))
    tick_labels = [df.index[pos].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_volume.set_xticks(tick_positions)
    ax_volume.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax_price.set_title(f"{symbol} Daily OHLC & Volume ({start} ~ {end})")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)

    ax_volume.set_ylabel("Volume")
    ax_volume.set_ylim(0, df["volume"].max() * 1.1)
    ax_volume.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """载入 qlib 数据，校验日期范围并输出图表。"""

    provider_path = DATA_DIR
    if not provider_path.exists():
        raise FileNotFoundError(
            f"QLib data directory not found: {provider_path}. Run build_us_dataset.py to create it."
        )

    init_qlib(provider_path)

    symbol = "TSM"
    start = "2025-01-01"
    end = "2025-09-29"

    coverage = get_instrument_coverage(symbol, provider_path)
    if coverage is not None:
        available_start, available_end = coverage
        requested_start, requested_end = pd.Timestamp(start), pd.Timestamp(end)
        if requested_end > available_end:
            raise ValueError(
                f"QLib dataset for {symbol} ends at {available_end.date()}, requested end {requested_end.date()} is beyond available range."
            )
        if requested_start < available_start:
            raise ValueError(
                f"QLib dataset for {symbol} starts at {available_start.date()}, requested start {requested_start.date()} is earlier than available range."
            )

    df = load_daily_frame(symbol, start, end)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} between {start} and {end}")

    plot_candlestick_with_volume(df, symbol, start, end)


if __name__ == "__main__":
    main()
