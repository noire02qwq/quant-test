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
    """计算指数移动平均线。"""

    return series.ewm(span=period, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """计算真实波动幅度，用于 ATR。"""

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


def _compute_kdj(df: pd.DataFrame, period: int = 9, smooth: int = 3) -> pd.DataFrame:
    """计算 KDJ 指标。"""

    low_min = df["low"].rolling(window=period, min_periods=1).min()
    high_max = df["high"].rolling(window=period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    rsv = (df["close"] - low_min) / denom * 100
    k = rsv.ewm(com=smooth - 1, adjust=False).mean()
    d = k.ewm(com=smooth - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """补充长线策略所需的技术指标。"""

    out = df.copy()
    for period in (5, 20, 60):
        out[f"ema_{period}"] = _ema(out["close"], period)
    ema_fast = _ema(out["close"], 12)
    ema_slow = _ema(out["close"], 26)
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    out["macd_dif"], out["macd_dea"], out["macd_hist"] = dif, dea, hist

    kdj = _compute_kdj(out)
    out = out.join(kdj)

    tr = _true_range(out)
    out["atr"] = tr.rolling(window=14, min_periods=14).mean()
    return out


def compute_regression_labels(df: pd.DataFrame) -> pd.Series:
    """依据 20 日观察窗口与止盈止损规则计算回归标签。"""

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
            # 未来数据不足 20 个交易日，标签跳过
            continue

        future_df = df.iloc[future_idx]
        stop_loss = entry_price - STOP_LOSS_MULT * atr
        stop_diff = STOP_LOSS_MULT * atr
        stop_gain = entry_price + STOP_GAIN_MULT * atr

        trigger = None
        for j, (_, future_row) in enumerate(future_df.iterrows()):
            if pd.isna(future_row[["high", "low"]]).any():
                continue
            if future_row["low"] <= stop_loss:
                trigger = "loss"
                break
            if future_row["high"] >= stop_gain:
                trigger = "gain"
                break

        if trigger == "loss":
            y_origin = -stop_diff
        elif trigger == "gain":
            y_origin = future_df["high"].max() - entry_price
        else:
            end_close = future_df.iloc[-1]["close"]
            if pd.isna(end_close):
                continue
            y_origin = end_close - entry_price

        label_value = y_origin / stop_diff if stop_diff != 0 else np.nan
        labels.iloc[i] = label_value

    return labels


def plot_diagnostics(df: pd.DataFrame) -> Path:
    """输出 2015 年的诊断图：K 线 + EMA、MACD、KDJ、回归标签。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = df.loc["2015-01-01":"2015-12-31"].dropna(subset=["ema_5", "ema_20", "ema_60"])
    if plot_df.empty:
        raise ValueError("2015 年数据为空，请先准备原始数据。")

    positions = np.arange(len(plot_df))
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]})

    ax_price, ax_macd, ax_kdj, ax_label = axes

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
    ax_price.set_title("TSM 2015 Daily K-line with EMA")
    ax_price.grid(True, linestyle="--", alpha=0.3)
    ax_price.legend(loc="upper left")

    # MACD 子图
    ax_macd.plot(positions, plot_df["macd_dif"], label="MACD DIF", color="#1f77b4")
    ax_macd.plot(positions, plot_df["macd_dea"], label="MACD DEA", color="#ff7f0e")
    ax_macd.bar(positions, plot_df["macd_hist"], label="Hist", color="#b22222", alpha=0.4)
    ax_macd.set_ylabel("MACD")
    ax_macd.grid(True, linestyle="--", alpha=0.3)
    ax_macd.legend(loc="upper left")

    # KDJ 子图
    ax_kdj.plot(positions, plot_df["kdj_k"], label="K", color="#1f77b4")
    ax_kdj.plot(positions, plot_df["kdj_d"], label="D", color="#ff7f0e")
    ax_kdj.plot(positions, plot_df["kdj_j"], label="J", color="#2ca02c")
    ax_kdj.set_ylabel("KDJ")
    ax_kdj.grid(True, linestyle="--", alpha=0.3)
    ax_kdj.legend(loc="upper left")

    # 标签子图
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
    output_path = OUTPUT_DIR / "tsm_2015_regression_label_diagnostics.png"
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
    """加载原始 CSV、计算指标与回归标签，并输出诊断图。"""

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"未找到原始数据文件：{RAW_FILE}，请先运行 prepare_data.py。")

    df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    enriched = compute_indicators(df)
    enriched["label_y"] = compute_regression_labels(enriched)

    dataset_path = save_dataset(enriched)
    plot_path = plot_diagnostics(enriched)
    valid_ratio = enriched["label_y"].notna().mean()
    print(f"标签数据已生成：{dataset_path}")
    print(f"诊断图输出：{plot_path}")
    print(f"有效样本占比：{valid_ratio:.2%}")


if __name__ == "__main__":
    main()
