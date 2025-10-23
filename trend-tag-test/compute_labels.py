from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trend_tag_analysis import (
    TrendScanParams,
    TREND_CLASS_MAP,
    assign_trend_classes,
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
    path = PROCESSED_DIR / "qqq_trend_classification_labels.csv"
    df.to_csv(path, index_label="date")
    return path


def compute_and_visualize(symbol: str = "QQQ") -> None:
    raw_path = RAW_DIR / f"{symbol}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"未找到原始数据文件：{raw_path}，请先运行 prepare_data.py。")

    price_df = pd.read_csv(raw_path, parse_dates=["date"]).sort_values("date").set_index("date")
    params = TrendScanParams()
    trend_df = compute_trend_scanning_labels(price_df, params)
    class_df, stats = assign_trend_classes(trend_df)
    merged = trend_df.join(class_df)

    dataset_path = save_dataset(merged)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    year_slice = merged.loc["2024-01-01":"2024-12-31"]
    if year_slice.dropna(subset=["trend_tvalue", "trend_ret"]).empty:
        raise ValueError("2024 年没有有效趋势样本，请检查数据范围。")
    plot_trend_timeseries(year_slice, symbol, OUTPUT_DIR / f"{symbol}_trend_2024.png")
    plot_histograms(merged, OUTPUT_DIR / f"{symbol}_trend_hist.png")

    total_valid = merged["trend_class_index"].notna().sum()
    ratios = {label: float((merged["trend_class"] == label).sum() / total_valid) for label in TREND_CLASS_MAP}

    print(f"[Done] 标签数据保存至 {dataset_path}")
    print(
        f"[Stats] thresholds: pos={stats['positive_threshold']:.4f}, "
        f"neg={stats['negative_threshold']:.4f}; ratios: "
        f"+1={ratios[1]:.3f}, 0={ratios[0]:.3f}, -1={ratios[-1]:.3f}"
    )
    print(f"[Plot] 2024 时序图: {OUTPUT_DIR / f'{symbol}_trend_2024.png'}")
    print(f"[Plot] 分布图: {OUTPUT_DIR / f'{symbol}_trend_hist.png'}")


def main() -> None:
    compute_and_visualize(symbol="QQQ")


if __name__ == "__main__":
    main()
