from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yfinance as yf

@dataclass
class TrendScanParams:
    min_horizon: int = 5
    max_horizon: int = 20


TREND_CLASS_MAP: Dict[int, int] = {-1: 0, 0: 1, 1: 2}


def fetch_price_data(symbol: str, start: str, end: str, cache_path: Path) -> pd.DataFrame:
    """Fetch daily bar data from Yahoo Finance and cache it locally."""

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if df.empty or set(df.columns) == {"QQQ"}:
            df = pd.DataFrame()
    else:
        data = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            actions=False,
        )
        if data.empty:
            raise RuntimeError(f"无法下载 {symbol} 的行情数据，请检查网络或时间范围。")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{col0}_{col1}" for col0, col1 in data.columns]
        data = data.reset_index().rename(columns={"Date": "date"})
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        for orig, target in rename_map.items():
            if orig in data.columns:
                data = data.rename(columns={orig: target})
            else:
                candidates = [col for col in data.columns if col.lower().startswith(orig.lower())]
                if candidates:
                    data = data.rename(columns={candidates[0]: target})
                else:
                    raise KeyError(f"下载数据中缺少列 {orig}")
        columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        data = data[columns]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path, index=False)
        df = data
    df = df[df.get("date").notna()]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("date").set_index("date")
    return df


def _linear_regression_tvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return slope and t-value of slope for simple linear regression."""

    n = len(x)
    if n < 3:
        return np.nan, np.nan
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if np.isclose(sxx, 0.0):
        return np.nan, np.nan
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    residuals = y - (intercept + slope * x)
    sse = np.sum(residuals**2)
    dof = n - 2
    if dof <= 0:
        return slope, np.nan
    mse = sse / dof
    std_err = np.sqrt(mse / sxx)
    if np.isclose(std_err, 0.0):
        t_value = np.sign(slope) * np.inf
    else:
        t_value = slope / std_err
    return slope, t_value


def compute_trend_scanning_labels(df: pd.DataFrame, params: TrendScanParams) -> pd.DataFrame:
    """Compute trend-scanning labels for the given DataFrame."""

    closes = df["adj_close"].fillna(df["close"]).values
    log_prices = np.log(closes)
    n = len(df)

    best_horizon = np.full(n, np.nan)
    best_tvalue = np.full(n, np.nan)
    best_slope = np.full(n, np.nan)
    best_ret = np.full(n, np.nan)
    best_target_index = np.full(n, -1, dtype=int)

    x_cache = {}

    for i in range(n):
        max_abs_t = -np.inf
        best_info = None
        for horizon in range(params.min_horizon, params.max_horizon + 1):
            j = i + horizon
            if j >= n:
                break
            length = horizon + 1
            if length not in x_cache:
                x_cache[length] = np.arange(length, dtype=float)
            x = x_cache[length]
            y = log_prices[i : j + 1]
            slope, t_value = _linear_regression_tvalue(x, y)
            if np.isnan(t_value):
                continue
            abs_t = abs(t_value)
            if abs_t > max_abs_t:
                max_abs_t = abs_t
                ret = float(closes[j] / closes[i] - 1)
                best_info = (horizon, j, slope, t_value, ret)
        if best_info is not None:
            horizon, j, slope, t_value, ret = best_info
            best_horizon[i] = horizon
            best_tvalue[i] = t_value
            best_slope[i] = slope
            best_ret[i] = ret
            best_target_index[i] = j

    result = df.copy()
    result["trend_window"] = best_horizon
    result["trend_slope"] = best_slope
    result["trend_tvalue"] = best_tvalue
    result["trend_ret"] = best_ret
    target_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_mask = best_target_index >= 0
    if valid_mask.any():
        target_indices = best_target_index[valid_mask].astype(int)
        target_dates[valid_mask] = df.index.to_numpy()[target_indices]
    result["trend_target_date"] = target_dates
    return result


def assign_trend_classes(
    df: pd.DataFrame,
    column: str = "trend_tvalue",
    positive_quantile: float = 2.0 / 3.0,
    negative_scale: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Assign {-1,0,1} classes based on trend t-value distribution."""

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")

    mask = df[column].notna()
    values = df.loc[mask, column].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"No valid values in column '{column}' for class assignment.")

    pos_threshold = np.quantile(values, positive_quantile)
    if pos_threshold <= 0:
        raise ValueError(
            "Positive threshold non-positive; cannot assign classes as specified. "
            "Check trend t-value distribution."
        )
    neg_threshold = -negative_scale * pos_threshold

    classes = np.full(len(df), np.nan)
    classes_indices = np.full(len(df), np.nan)

    positive_mask = mask & (df[column] >= pos_threshold)
    negative_mask = mask & (df[column] < neg_threshold)
    neutral_mask = mask & ~(positive_mask | negative_mask)

    classes[positive_mask] = 1
    classes[negative_mask] = -1
    classes[neutral_mask] = 0

    for class_value, class_index in TREND_CLASS_MAP.items():
        idx_mask = classes == class_value
        classes_indices[idx_mask] = class_index

    result = pd.DataFrame(
        {
            "trend_class": classes,
            "trend_class_index": classes_indices,
        },
        index=df.index,
    )

    base_count = mask.sum()
    if base_count == 0:
        raise ValueError("No valid samples available for class statistics.")

    stats = {
        "positive_threshold": float(pos_threshold),
        "negative_threshold": float(neg_threshold),
        "positive_ratio": float(positive_mask.sum() / base_count),
        "negative_ratio": float(negative_mask.sum() / base_count),
        "neutral_ratio": float(neutral_mask.sum() / base_count),
    }
    return result, stats


def _highlight_segments(ax: plt.Axes, indicator: pd.Series, positive_color="#ffaaaa", negative_color="#a4f4a4") -> None:
    """Highlight contiguous segments based on indicator sign."""

    values = indicator.fillna(0).values
    start = None
    current_sign = None
    for idx, val in enumerate(values):
        sign = 1 if val > 0 else (-1 if val < 0 else 0)
        if sign == 0:
            if start is not None:
                color = positive_color if current_sign > 0 else negative_color
                ax.axvspan(start - 0.5, idx - 0.5, color=color, alpha=0.2)
                start = None
                current_sign = None
            continue
        if start is None:
            start = idx
            current_sign = sign
        elif sign != current_sign:
            color = positive_color if current_sign > 0 else negative_color
            ax.axvspan(start - 0.5, idx - 0.5, color=color, alpha=0.2)
            start = idx
            current_sign = sign
    if start is not None:
        color = positive_color if current_sign > 0 else negative_color
        ax.axvspan(start - 0.5, len(values) - 0.5, color=color, alpha=0.2)


def _highlight_class_segments(ax: plt.Axes, class_series: pd.Series) -> None:
    values = class_series.fillna(0).to_numpy(dtype=int)
    start = None
    current = None
    color_map = {1: "#86efac", -1: "#fca5a5"}
    for idx, val in enumerate(values):
        if val == 0:
            if start is not None and current in color_map:
                ax.axvspan(start - 0.5, idx - 0.5, color=color_map[current], alpha=0.2)
                start = None
                current = None
            continue
        if start is None:
            start = idx
            current = val
        elif val != current:
            if current in color_map:
                ax.axvspan(start - 0.5, idx - 0.5, color=color_map[current], alpha=0.2)
            start = idx
            current = val
    if start is not None and current in color_map:
        ax.axvspan(start - 0.5, len(values) - 0.5, color=color_map[current], alpha=0.2)


def plot_trend_timeseries(df: pd.DataFrame, symbol: str, output_path: Path) -> None:
    df_plot = df.dropna(subset=["trend_tvalue", "trend_ret"])
    if df_plot.empty:
        raise ValueError("缺少用于绘图的数据，请检查趋势标签计算是否成功。")

    positions = np.arange(len(df_plot))

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    ax_price, ax_tvalue, ax_ret = axes

    candle_width = 0.6
    for idx, (_, row) in enumerate(df_plot.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
        ax_price.add_patch(candle)

    if "trend_class" in df_plot.columns:
        _highlight_class_segments(ax_price, df_plot["trend_class"])
    else:
        _highlight_segments(ax_price, df_plot["trend_tvalue"])

    ax_price.set_title(f"{symbol} 2024 Candlestick with Trend Classes")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)

    ax_tvalue.plot(positions, df_plot["trend_tvalue"], color="#1f77b4", label="t-value")
    ax_tvalue.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
    ax_tvalue.set_ylabel("Trend t-value")
    ax_tvalue.grid(True, linestyle="--", alpha=0.3)
    ax_tvalue.legend(loc="upper left")

    ax_ret.plot(positions, df_plot["trend_ret"], color="#d62728", label="return")
    ax_ret.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
    ax_ret.set_ylabel("Return")
    ax_ret.set_xlabel("Date")
    ax_ret.grid(True, linestyle="--", alpha=0.3)
    ax_ret.legend(loc="upper left")

    tick_step = max(len(df_plot) // 10, 1)
    tick_positions = list(range(0, len(df_plot), tick_step))
    if tick_positions[-1] != len(df_plot) - 1:
        tick_positions.append(len(df_plot) - 1)
    tick_labels = [df_plot.index[pos].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_ret.set_xticks(tick_positions)
    ax_ret.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _add_tercile_annotations(ax: plt.Axes, data: pd.Series) -> None:
    data = data.dropna()
    if data.empty:
        return
    q1, q2 = np.quantile(data, [1 / 3, 2 / 3])
    ylim = ax.get_ylim()
    for q, color in zip((q1, q2), ("#2f4f4f", "#7f0000")):
        ax.axvline(q, color=color, linestyle="--", linewidth=1)
        ax.text(
            q,
            ylim[1] * 0.95,
            f"{q:.4f}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=9,
            color=color,
            backgroundcolor="white",
        )
    ax.set_ylim(ylim)


def plot_histograms(df: pd.DataFrame, output_path: Path) -> None:
    df_all = df.dropna(subset=["trend_tvalue", "trend_ret"])
    df_2024 = df_all.loc["2024-01-01":"2024-12-31"]

    if df_all.empty or df_2024.empty:
        raise ValueError("用于直方图的数据为空，请检查日期范围或标签计算。")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].hist(df_2024["trend_tvalue"], bins=30, color="#1f77b4", alpha=0.8)
    axes[0].set_title("2024 Trend t-value Distribution")
    axes[0].set_xlabel("Trend t-value")
    axes[0].set_ylabel("Frequency")
    _add_tercile_annotations(axes[0], df_2024["trend_tvalue"])

    axes[1].hist(df_all["trend_tvalue"], bins=30, color="#ff7f0e", alpha=0.8)
    axes[1].set_title("2014-2024 Trend t-value Distribution")
    axes[1].set_xlabel("Trend t-value")
    axes[1].set_ylabel("Frequency")
    _add_tercile_annotations(axes[1], df_all["trend_tvalue"])

    axes[2].hist(df_2024["trend_ret"], bins=30, color="#2ca02c", alpha=0.8)
    axes[2].set_title("2024 Trend Return Distribution")
    axes[2].set_xlabel("Trend Return")
    axes[2].set_ylabel("Frequency")
    _add_tercile_annotations(axes[2], df_2024["trend_ret"])

    axes[3].hist(df_all["trend_ret"], bins=30, color="#d62728", alpha=0.8)
    axes[3].set_title("2014-2024 Trend Return Distribution")
    axes[3].set_xlabel("Trend Return")
    axes[3].set_ylabel("Frequency")
    _add_tercile_annotations(axes[3], df_all["trend_ret"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trend-scanning label computation and visualization.")
    parser.add_argument(
        "--symbols",
        default="QQQ",
        help="Comma-separated list of tickers (default: QQQ).",
    )
    parser.add_argument("--start", default="2013-12-01", help="Start date (inclusive)")
    parser.add_argument("--end", default="2025-01-31", help="End date (inclusive)")
    parser.add_argument("--min-window", type=int, default=5, help="Minimum observation window (days)")
    parser.add_argument("--max-window", type=int, default=20, help="Maximum observation window (days)")
    parser.add_argument("--output-dir", default="trend-tag-test/outputs", help="Output directory")
    parser.add_argument("--cache-dir", default=None, help="Optional cache directory for raw CSVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    params = TrendScanParams(min_horizon=args.min_window, max_horizon=args.max_window)

    for symbol in symbols:
        cache_path = cache_dir / f"{symbol}_daily.csv"
        price_df = fetch_price_data(symbol, args.start, args.end, cache_path)
        labels_df = compute_trend_scanning_labels(price_df, params)
        class_df, class_stats = assign_trend_classes(labels_df)
        labels_df = labels_df.join(class_df)

        labels_csv = output_dir / f"{symbol}_trend_labels.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(labels_csv)

        labels_2024 = labels_df.loc["2024-01-01":"2024-12-31"]
        if labels_2024.dropna(subset=["trend_tvalue", "trend_ret"]).empty:
            print(f"[Warn] {symbol} has no valid trend labels within 2024; skipping time series plot.")
        else:
            plot_trend_timeseries(labels_2024, symbol, output_dir / f"{symbol}_trend_2024.png")
        plot_histograms(labels_df, output_dir / f"{symbol}_trend_hist.png")

        print(f"[Done] Saved label data to {labels_csv}")
        print(f"[Done] Saved 2024 time series plot to {output_dir / f'{symbol}_trend_2024.png'}")
        print(f"[Done] Saved histogram plot to {output_dir / f'{symbol}_trend_hist.png'}")
        print(
            f"[Stats] {symbol} thresholds: pos={class_stats['positive_threshold']:.4f}, "
            f"neg={class_stats['negative_threshold']:.4f}; ratios (pos/neu/neg)="
            f"{class_stats['positive_ratio']:.3f}/{class_stats['neutral_ratio']:.3f}/{class_stats['negative_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
