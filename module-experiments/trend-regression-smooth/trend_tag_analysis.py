from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class TrendScanParams:
    min_horizon: int = 5
    max_horizon: int = 15
    smoothing_span: int | None = 5


def fetch_price_data(symbol: str, start: str, end: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if df.empty or set(df.columns) == {symbol}:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
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
            raise RuntimeError(f"无法下载 {symbol} 的行情数据，请检查参数。")
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
        data = data[["date", "open", "high", "low", "close", "adj_close", "volume"]]
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


def _linear_regression_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if np.isclose(sxx, 0.0):
        return np.nan, np.nan, np.nan
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    residuals = y - (intercept + slope * x)
    sse = np.sum(residuals**2)
    sst = np.sum((y - y_mean) ** 2)
    r2 = 1 - sse / (sst + 1e-12)
    dof = n - 2
    if dof <= 0:
        return slope, np.nan, r2
    mse = sse / dof
    std_err = np.sqrt(mse / sxx)
    if np.isclose(std_err, 0.0):
        t_value = np.sign(slope) * np.inf
    else:
        t_value = slope / std_err
    return slope, t_value, r2


def compute_trend_scanning_labels(df: pd.DataFrame, params: TrendScanParams) -> pd.DataFrame:
    closes = df["adj_close"].fillna(df["close"]).astype(float)
    span = params.smoothing_span or 0
    if span and span > 1:
        price_series = closes.ewm(span=span, adjust=False).mean()
    else:
        price_series = closes
    log_prices = np.log(price_series)
    prices = price_series.values
    log_values = log_prices.values
    n = len(df)

    best_horizon = np.full(n, np.nan)
    best_tvalue = np.full(n, np.nan)
    best_slope = np.full(n, np.nan)
    best_ret_pct = np.full(n, np.nan)
    best_r2 = np.full(n, np.nan)
    best_target_index = np.full(n, -1, dtype=int)

    x_cache: dict[int, np.ndarray] = {}
    date_values = df.index.to_numpy()

    for i in range(n):
        base_price = prices[i]
        if not np.isfinite(base_price) or not np.isfinite(log_values[i]):
            continue
        max_abs_t = -np.inf
        best_info = None
        for horizon in range(params.min_horizon, params.max_horizon + 1):
            j = i + horizon
            if j >= n:
                break
            target_price = prices[j]
            segment = log_values[i : j + 1]
            if not np.isfinite(target_price) or not np.isfinite(segment).all():
                continue
            length = horizon + 1
            if length not in x_cache:
                x_cache[length] = np.arange(length, dtype=float)
            x = x_cache[length]
            slope, t_value, r2 = _linear_regression_stats(x, segment)
            if np.isnan(t_value):
                continue
            abs_t = abs(t_value)
            if abs_t > max_abs_t:
                ret_pct = float((target_price / base_price - 1.0) * 100.0)
                best_info = (horizon, j, slope, t_value, ret_pct, r2)
                max_abs_t = abs_t
        if best_info is not None:
            horizon, j, slope, t_value, ret_pct, r2 = best_info
            best_horizon[i] = horizon
            best_tvalue[i] = t_value
            best_slope[i] = slope
            best_ret_pct[i] = ret_pct
            best_r2[i] = r2
            best_target_index[i] = j

    result = df.copy()
    result["trend_window"] = best_horizon
    result["trend_slope"] = best_slope
    result["trend_tvalue"] = best_tvalue
    result["trend_ret_pct"] = best_ret_pct
    result["trend_r2"] = best_r2
    target_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_mask = best_target_index >= 0
    if valid_mask.any():
        target_indices = best_target_index[valid_mask].astype(int)
        target_dates[valid_mask] = date_values[target_indices]
    result["trend_target_date"] = target_dates
    return result


def _highlight_segments(
    ax: plt.Axes,
    indicator: pd.Series,
    positive_color="#ffaaaa",
    negative_color="#a4f4a4",
    quality: pd.Series | None = None,
    min_quality: float = 0.0,
) -> None:
    values = indicator.fillna(0).values
    quality_values = None
    if quality is not None:
        quality_values = quality.reindex(indicator.index).fillna(0.0).values
    start = None
    current_sign = None
    for idx, val in enumerate(values):
        sign = 1 if val > 0 else (-1 if val < 0 else 0)
        if quality_values is not None:
            if idx >= len(quality_values) or quality_values[idx] < min_quality or np.isnan(quality_values[idx]):
                sign = 0
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


def plot_trend_timeseries(
    ema_df: pd.DataFrame,
    close_df: pd.DataFrame,
    symbol: str,
    output_path: Path,
    min_r2: float = 0.75,
) -> None:
    variants = [
        ("EMA5 (log price)", ema_df),
        ("Close (log price)", close_df),
    ]
    prepared: List[Tuple[str, pd.DataFrame]] = []
    for name, raw in variants:
        if raw is None:
            continue
        df_plot = raw.dropna(subset=["trend_tvalue", "trend_ret_pct"])
        if df_plot.empty:
            continue
        prepared.append((name, df_plot))

    if not prepared:
        raise ValueError("缺少用于绘图的数据，请检查趋势标签计算是否成功。")

    num_cols = len(prepared)
    fig, axes = plt.subplots(
        5,
        num_cols,
        figsize=(9.5 * num_cols, 12),
        sharex="col",
        gridspec_kw={"height_ratios": [3.0, 1.2, 1.2, 1.0, 1.0]},
    )
    if num_cols == 1:
        axes = np.array(axes).reshape(5, 1)

    candle_width = 0.6
    for col, (variant_name, df_plot) in enumerate(prepared):
        ax_price = axes[0, col]
        ax_tvalue = axes[1, col]
        ax_ret = axes[2, col]
        ax_window = axes[3, col]
        ax_r2 = axes[4, col]

        positions = np.arange(len(df_plot))
        for idx, (_, row) in enumerate(df_plot.iterrows()):
            color = "red" if row["close"] >= row["open"] else "green"
            ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
            lower = min(row["open"], row["close"])
            height = max(row["open"], row["close"]) - lower or 1e-10
            candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
            ax_price.add_patch(candle)

        _highlight_segments(
            ax_price,
            df_plot["trend_tvalue"],
            quality=df_plot.get("trend_r2"),
            min_quality=min_r2,
        )
        ax_price.set_title(f"{symbol} - {variant_name}")
        ax_price.set_ylabel("Price")
        ax_price.grid(True, linestyle="--", alpha=0.3)

        ax_tvalue.plot(positions, df_plot["trend_tvalue"], color="#1f77b4", label="t-value")
        ax_tvalue.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        ax_tvalue.set_ylabel("Trend t-value")
        ax_tvalue.grid(True, linestyle="--", alpha=0.3)
        ax_tvalue.legend(loc="upper left")

        ax_ret.plot(positions, df_plot["trend_ret_pct"], color="#d62728", label="Return (%)")
        ax_ret.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        ax_ret.set_ylabel("Return (%)")
        ax_ret.grid(True, linestyle="--", alpha=0.3)
        ax_ret.legend(loc="upper left")

        ax_window.plot(positions, df_plot["trend_window"], color="#9467bd", label="Horizon (days)")
        ax_window.set_ylabel("Days")
        ax_window.grid(True, linestyle="--", alpha=0.3)
        ax_window.legend(loc="upper left")

        ax_r2.plot(positions, df_plot["trend_r2"], color="#8c564b", label="R²")
        ax_r2.set_ylabel("R²")
        ax_r2.set_xlabel("Date")
        ax_r2.grid(True, linestyle="--", alpha=0.3)
        ax_r2.set_ylim(0.0, 1.05)
        ax_r2.legend(loc="upper left")

        tick_step = max(len(df_plot) // 10, 1)
        tick_positions = list(range(0, len(df_plot), tick_step))
        if tick_positions[-1] != len(df_plot) - 1:
            tick_positions.append(len(df_plot) - 1)
        tick_labels = [df_plot.index[pos].strftime("%Y-%m-%d") for pos in tick_positions]
        ax_r2.set_xticks(tick_positions)
        ax_r2.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.suptitle(f"{symbol} Trend-Scanning Comparison", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
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
        ax.text(q, ylim[1] * 0.95, f"{q:.4f}", rotation=90, ha="right", va="top", fontsize=9, color=color, backgroundcolor="white")
    ax.set_ylim(ylim)


def plot_histograms(df: pd.DataFrame, output_path: Path) -> None:
    df_all = df.dropna(subset=["trend_tvalue", "trend_ret_pct"])
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

    axes[2].hist(df_2024["trend_ret_pct"], bins=30, color="#2ca02c", alpha=0.8)
    axes[2].set_title("2024 Trend Return Distribution (%)")
    axes[2].set_xlabel("Trend Return (%)")
    axes[2].set_ylabel("Frequency")
    _add_tercile_annotations(axes[2], df_2024["trend_ret_pct"])

    axes[3].hist(df_all["trend_ret_pct"], bins=30, color="#d62728", alpha=0.8)
    axes[3].set_title("2014-2024 Trend Return Distribution (%)")
    axes[3].set_xlabel("Trend Return (%)")
    axes[3].set_ylabel("Frequency")
    _add_tercile_annotations(axes[3], df_all["trend_ret_pct"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trend-scanning label computation and visualization (regression focus).")
    parser.add_argument("--symbols", default="QQQ", help="Comma-separated tickers (default: QQQ)")
    parser.add_argument("--start", default="2013-12-01", help="Start date")
    parser.add_argument("--end", default="2025-01-31", help="End date")
    parser.add_argument("--min-window", type=int, default=5, help="Minimum observation window")
    parser.add_argument("--max-window", type=int, default=20, help="Maximum observation window")
    parser.add_argument("--output-dir", default="module-experiments/trend-regression-test/outputs", help="Output directory")
    parser.add_argument("--cache-dir", default=None, help="Optional cache directory")
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
        raw_params = TrendScanParams(
            min_horizon=params.min_horizon,
            max_horizon=params.max_horizon,
            smoothing_span=0,
        )
        labels_df_close = compute_trend_scanning_labels(price_df, raw_params)

        labels_csv = output_dir / f"{symbol}_trend_regression.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(labels_csv)

        labels_2024 = labels_df.loc["2024-01-01":"2024-12-31"]
        labels_close_2024 = labels_df_close.loc["2024-01-01":"2024-12-31"]
        if (
            labels_2024.dropna(subset=["trend_tvalue", "trend_ret_pct"]).empty
            and labels_close_2024.dropna(subset=["trend_tvalue", "trend_ret_pct"]).empty
        ):
            print(f"[Warn] {symbol} has no valid trend samples in 2024; skipping timeseries plot.")
        else:
            plot_trend_timeseries(labels_2024, labels_close_2024, symbol, output_dir / f"{symbol}_trend_2024.png")
        plot_histograms(labels_df, output_dir / f"{symbol}_trend_hist.png")

        print(f"[Done] Saved label data to {labels_csv}")
        print(f"[Done] Saved 2024 time series plot to {output_dir / f'{symbol}_trend_2024.png'}")
        print(f"[Done] Saved histogram plot to {output_dir / f'{symbol}_trend_hist.png'}")


if __name__ == "__main__":
    main()
