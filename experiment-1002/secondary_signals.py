"""结合一级指标计算二级买入信号，并在图表中高亮对应日期。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_US

# 将 scripts 目录加入搜索路径，复用已有一级指标实现
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


OUTPUT_DIR = REPO_ROOT / "outputs"


def _cross_over(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """判断 A 上穿 B（含边界），返回布尔序列。"""

    cond = (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))
    return cond.fillna(False)


def compute_secondary_signals(df: pd.DataFrame) -> pd.DataFrame:
    """根据一级指标结果生成二级买入信号列。"""

    signals = pd.DataFrame(index=df.index)

    # MACD 金叉 + 柱状图为正
    macd_cross = _cross_over(df["macd_dif"], df["macd_dea"]) & (df["macd_hist"] > 0)
    signals["signal_macd"] = macd_cross

    # KDJ 金叉 + K <= 70
    kdj_cross = _cross_over(df["kdj_k"], df["kdj_d"]) & (df["kdj_k"] <= 70)
    signals["signal_kdj"] = kdj_cross

    # EMA20 > EMA60 且当日阳线向上穿越 EMA20
    ema20 = df["ema_20"]
    ema60 = df["ema_60"]
    ema_cross = (
        (ema20 > ema60)
        & (df["close"] > df["open"])
        & (df["close"] >= ema20)
        & (df["open"] <= ema20)
        & (df["close"].shift(1) <= ema20.shift(1))
    )
    signals["signal_ema"] = ema_cross

    # SAR 从 K 线上方翻转至下方
    sar_prev_above = df["sar"].shift(1) > df[["open", "close"]].shift(1).max(axis=1)
    sar_now_below = df["sar"] <= df[["open", "close"]].min(axis=1)
    signals["signal_sar"] = sar_prev_above & sar_now_below

    # DMI：PDI 金叉 MDI 或 ADX 金叉 ADXR
    pdi_cross = _cross_over(df["pdi"], df["mdi"])
    adx_cross = _cross_over(df["adx"], df["adxr"])
    signals["signal_dmi"] = pdi_cross | adx_cross

    # ADTM 金叉 + ADTM < 0.5
    adtm_cross = _cross_over(df["adtm"], df["adtmma"]) & (df["adtm"] < 0.5)
    signals["signal_adtm"] = adtm_cross

    # DDI 由负变正（x-1 < 0, x+1 > 0）
    ddi_series = df["ddi"]
    ddi_cross = (ddi_series.shift(1) < 0) & (ddi_series.shift(-1) > 0)
    signals["signal_ddi"] = ddi_cross

    # DPO 由负变正
    dpo_series = df["dpo"]
    dpo_cross = (dpo_series.shift(1) < 0) & (dpo_series.shift(-1) > 0)
    signals["signal_dpo"] = dpo_cross

    # OSC 上穿 OSCEMA
    osc_cross = _cross_over(df["osc"], df["osc_signal"])
    signals["signal_osc"] = osc_cross

    # SRMI 由负变正
    srmi_series = df["srmi"]
    srmi_cross = (srmi_series.shift(1) < 0) & (srmi_series.shift(-1) > 0)
    signals["signal_srmi"] = srmi_cross

    return pd.concat([df, signals], axis=1)


def _find_segments(signal: pd.Series) -> List[Tuple[int, int]]:
    """将布尔信号拆分为若干连续区间。"""

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


def plot_with_signals(df: pd.DataFrame, ema_columns: Iterable[str]) -> Tuple[plt.Figure, plt.Axes, List[Rectangle]]:
    """绘制 K 线与 EMA，并以半透明背景标注二级信号。"""

    required_cols = list(ema_columns) + ["open", "high", "low", "close"]
    df_plot = df.dropna(subset=required_cols).copy()
    if df_plot.empty:
        raise ValueError("有效数据为空，无法绘图，请检查时间区间或指标窗口。")

    signal_styles: Dict[str, Tuple[str, str]] = {
        "signal_macd": ("MACD Golden Cross", "#f94144"),
        "signal_kdj": ("KDJ Golden Cross", "#577590"),
        "signal_ema": ("EMA Bullish Break", "#f3722c"),
        "signal_sar": ("SAR Flip", "#43aa8b"),
        "signal_dmi": ("DMI Cross", "#90be6d"),
        "signal_adtm": ("ADTM Break", "#f9c74f"),
        "signal_ddi": ("DDI Positive", "#277da1"),
        "signal_dpo": ("DPO Positive", "#7209b7"),
        "signal_osc": ("OSC Cross", "#ff9f1c"),
        "signal_srmi": ("SRMI Positive", "#2ec4b6"),
    }

    positions = np.arange(len(df_plot))
    fig, ax_price = plt.subplots(figsize=(14, 6))

    # 绘制蜡烛图
    candle_width = 0.6
    for idx, (_, row) in zip(positions, df_plot.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
        ax_price.add_patch(candle)

    # 绘制 EMA 曲线
    for ema_col in ema_columns:
        ax_price.plot(positions, df_plot[ema_col], label=ema_col.upper())

    ax_price.set_title("TSM Daily OHLC with Secondary Signals")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)

    # 高亮二级信号
    highlight_handles: List[Rectangle] = []
    for signal_name, (label, color) in signal_styles.items():
        if signal_name not in df_plot.columns:
            continue
        segments = _find_segments(df_plot[signal_name].fillna(False))
        if not segments:
            continue
        handle = Rectangle((0, 0), 1, 1, color=color, alpha=0.18)
        handle.set_label(label)
        highlight_handles.append(handle)
        for start, end in segments:
            ax_price.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.18, zorder=0)

    # 设置 X 轴刻度
    tick_step = max(len(df_plot) // 10, 1)
    tick_positions = list(range(0, len(df_plot), tick_step))
    if tick_positions[-1] != len(df_plot) - 1:
        tick_positions.append(len(df_plot) - 1)
    tick_labels = [df_plot.index[min(pos, len(df_plot) - 1)].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_price.set_xticks(tick_positions)
    ax_price.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    return fig, ax_price, highlight_handles


def save_chart_and_legend(fig: plt.Figure, ax: plt.Axes, handles: List[Rectangle], outputdir: Path) -> None:
    """保存主图和图例分别为 png。"""

    outputdir.mkdir(parents=True, exist_ok=True)

    chart_path = outputdir / "secondary_signals_chart.png"
    fig.savefig(chart_path, dpi=150)

    legend_fig, legend_ax = plt.subplots(figsize=(3.5, 4.5))
    legend_ax.axis("off")
    ema_handles, ema_labels = ax.get_legend_handles_labels()
    legend_handles = ema_handles + handles
    legend_labels = ema_labels + [h.get_label() for h in handles]
    legend_ax.legend(legend_handles, legend_labels, loc="center", frameon=False)
    legend_fig.tight_layout()
    legend_path = outputdir / "secondary_signals_legend.png"
    legend_fig.savefig(legend_path, dpi=150)
    plt.close(legend_fig)


def main() -> None:
    """生成 TSM 指标并输出二级信号图与图例。"""

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"未找到 qlib 数据目录：{DATA_DIR}，请先运行 build_us_dataset.py 构建数据。"
        )

    qlib.init(provider_uri=str(DATA_DIR), region=REG_US)

    symbol = "TSM"
    start = "2024-01-01"
    end = "2025-09-30"

    price_df = fetch_price_data(symbol, start, end)
    calculator = TechnicalIndicatorCalculator(price_df, IndicatorParams())
    indicator_df = calculator.compute()
    full_df = compute_secondary_signals(indicator_df)

    ema_cols = [f"ema_{p}" for p in calculator.params.ema_periods]
    fig, ax_price, highlight_handles = plot_with_signals(full_df, ema_cols)
    save_chart_and_legend(fig, ax_price, highlight_handles, OUTPUT_DIR)

    plt.close(fig)


if __name__ == "__main__":
    main()
