from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trend_tag_analysis import (
    TrendScanParams,
    compute_trend_scanning_labels,
    plot_histograms,
    plot_trend_timeseries,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXPERIMENT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"


def save_dataset(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "qqq_trend_regression_smooth_labels.csv"
    df.to_csv(path, index_label="date")
    return path


def compute_and_visualize(symbol: str = "QQQ") -> None:
    raw_path = RAW_DIR / f"{symbol}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"未找到原始数据文件：{raw_path}，请先运行 prepare_data.py。")

    price_df = pd.read_csv(raw_path, parse_dates=["date"]).sort_values("date").set_index("date")
    params = TrendScanParams()
    trend_df = compute_trend_scanning_labels(price_df, params)
    params_close = TrendScanParams(min_horizon=params.min_horizon, max_horizon=params.max_horizon, smoothing_span=0)
    trend_df_close = compute_trend_scanning_labels(price_df, params_close)

    dataset_path = save_dataset(trend_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    year_slice = trend_df.loc["2024-01-01":"2024-12-31"]
    year_slice_close = trend_df_close.loc["2024-01-01":"2024-12-31"]
    if year_slice.dropna(subset=["trend_tvalue", "trend_ret_pct"]).empty and year_slice_close.dropna(
        subset=["trend_tvalue", "trend_ret_pct"]
    ).empty:
        raise ValueError("2024 年没有有效趋势样本，请检查数据范围。")
    plot_trend_timeseries(year_slice, year_slice_close, symbol, OUTPUT_DIR / f"{symbol}_trend_2024.png")
    plot_histograms(trend_df, OUTPUT_DIR / f"{symbol}_trend_hist.png")

    valid_ratio = trend_df["trend_tvalue"].notna().mean()
    print(f"[Done] 标签数据保存至 {dataset_path}")
    print(f"[Info] trend_tvalue 有效占比：{valid_ratio:.2%}")
    print(f"[Plot] 2024 时序图: {OUTPUT_DIR / f'{symbol}_trend_2024.png'}")
    print(f"[Plot] 分布图: {OUTPUT_DIR / f'{symbol}_trend_hist.png'}")


def main() -> None:
    compute_and_visualize(symbol="QQQ")


if __name__ == "__main__":
    main()
