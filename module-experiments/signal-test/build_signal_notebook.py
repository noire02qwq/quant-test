from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import nbformat as nbf


def build_cells() -> List[Tuple[str, str]]:
    """Return an ordered list of (cell_type, content) tuples for the notebook."""

    markdown_intro = """# 信号测试分析工作簿

本文档按照 `module-experiments/signal-test/draft.md` 的要求，构建从数据准备、技术指标与信号计算、三围栏标签评估到统计报告导出的完整流程。所有代码按步骤拆分，并在关键环节穿插说明。"""

    markdown_overview = """## 流程概览
- 加载 2020-01-01 至 2025-08-31 期间的标的池行情，并预留 60 日缓冲
- 计算 KDJ、MACD、EMA、DMI、ATR 等技术指标，统一生成信号
- 定义 “t 日上穿且 t-2、t-3 日位于下方” 的上穿判定
- 在 horizon=5~15、ATR 倍数=1.0~3.0 的网格上执行三围栏标签
- 汇总每个信号的胜率、净胜次数，并绘制示例图与热力图
- 导出包含单信号章节与全局对比表格的 HTML 报告"""

    code_imports = """from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import display, HTML

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.25,
})"""

    markdown_universe = """## 标的池与基础参数
使用 draft 中给出的行业划分构建标的池，并声明核心时间、网格与文件路径参数。"""

    code_parameters = """def locate_data_dir() -> Path:
    search_roots = [Path.cwd()]
    # 向上两级以兼容 Notebook 位于子目录的情况
    search_roots.extend(path for path in Path.cwd().parents[:3])
    for root in search_roots:
        candidate = root / "data/qlib_us_selected/source"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("未找到 data/qlib_us_selected/source 目录，请确认数据是否已下载。")
DATA_DIR = locate_data_dir()
PROJECT_ROOT = DATA_DIR.parents[2]
OUTPUT_DIR = (PROJECT_ROOT / "module-experiments" / "signal-test").resolve()
ASSET_DIR = OUTPUT_DIR / "report_assets"
HTML_REPORT_PATH = OUTPUT_DIR / "signal_report.html"

ANALYSIS_START = pd.Timestamp("2020-01-01")
ANALYSIS_END = pd.Timestamp("2025-08-31")
BUFFER_DAYS = 60
PLOT_START = pd.Timestamp("2024-01-01")
PLOT_END = pd.Timestamp("2024-12-31")
HORIZONS = list(range(5, 16, 2))
ATR_MULTIPLIERS = [round(x, 1) for x in np.linspace(1.0, 3.0, num=11)]
REWARD_RATIO = 1.5
MAX_HORIZON = max(HORIZONS)

TICKER_GROUPS = {
    "科技": [
        "NVDA", "TSM", "AAPL", "MSFT", "GOOGL", "AMZN", "SAP", "ANET", "AVGO", "IBM",
        "TSLA", "PLTR", "ADP", "ALAB", "ADI", "TXN", "MU", "QCOM", "ARM", "SNDK", "RELX",
    ],
    "金融": [
        "JPM", "MS", "KKR", "MAIN", "V", "AXP", "PGR", "ICE", "BN", "SPGI", "BX", "NDAQ", "ARES", "STT",
    ],
    "传统消费与工业": ["ABBV", "CAT", "RTX", "VST", "MNST", "MCD", "CVX", "HWM"],
    "ADR": ["DBSDY", "ABBNY", "TKOMY", "TOELY", "NTDOY", "SFTBY"],
}

available_files = {path.stem for path in DATA_DIR.glob("*.csv")}
active_symbols: List[str] = []
missing_symbols: List[str] = []
for group, tickers in TICKER_GROUPS.items():
    for ticker in tickers:
        if ticker in available_files:
            active_symbols.append(ticker)
        else:
            missing_symbols.append(ticker)
active_symbols = sorted(set(active_symbols))

print(f"数据目录: {DATA_DIR}")
print(f"可用标的: {len(active_symbols)} 个 / draft 总计 {sum(len(v) for v in TICKER_GROUPS.values())} 个")
if missing_symbols:
    print("未找到的标的:", ", ".join(missing_symbols))"""

    markdown_load = """## 数据读取
从 CSV 中读取日频 OHLCV 数据，同时预留前后缓冲期以保证技术指标和标签的滚动窗口完整。"""

    code_load_fn = """def load_price_data(symbols: Iterable[str], start: pd.Timestamp, end: pd.Timestamp, buffer_days: int) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    start_with_buffer = start - pd.Timedelta(days=buffer_days)
    end_with_buffer = end + pd.Timedelta(days=buffer_days)
    for symbol in symbols:
        path = DATA_DIR / f"{symbol}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        df = df.loc[start_with_buffer:end_with_buffer, ["open", "high", "low", "close", "volume"]]
        frames[symbol] = df
    return frames"""

    code_load_run = """raw_data = load_price_data(active_symbols, ANALYSIS_START, ANALYSIS_END, BUFFER_DAYS)
print(f"成功载入 {len(raw_data)} 只标的的数据。")
sample_symbol = next(iter(raw_data))
print(f"示例标的: {sample_symbol}")
display(raw_data[sample_symbol].head())"""

    markdown_indicators = """## 技术指标计算
复用常见公式生成 EMA、MACD、KDJ、DMI、ATR 等指标，为后续信号判定准备输入。"""

    code_indicators = """@dataclass
class IndicatorParams:
    ema_periods: Tuple[int, ...] = (5, 10, 20, 60)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    kdj_period: int = 9
    kdj_smooth: int = 3
    dmi_period: int = 14
    atr_period: int = 14


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": hist})


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


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(df).rolling(window=period).mean()


def _dmi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    tr = _true_range(df)
    tr_sum = tr.rolling(window=period).sum()
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = pd.Series(plus_dm, index=df.index).rolling(window=period).sum() * 100 / tr_sum
    minus_di = pd.Series(minus_dm, index=df.index).rolling(window=period).sum() * 100 / tr_sum
    dx = (plus_di - minus_di).abs() * 100 / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    adxr = (adx + adx.shift(period)) / 2
    return pd.DataFrame({"pdi": plus_di, "mdi": minus_di, "adx": adx, "adxr": adxr})


def _kdj(df: pd.DataFrame, period: int, smooth: int) -> pd.DataFrame:
    low_min = df["low"].rolling(window=period).min()
    high_max = df["high"].rolling(window=period).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(alpha=1 / smooth, adjust=False).mean()
    d = k.ewm(alpha=1 / smooth, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})


def compute_indicator_panel(df: pd.DataFrame, params: IndicatorParams | None = None) -> pd.DataFrame:
    params = params or IndicatorParams()
    out = df.copy()
    for period in params.ema_periods:
        out[f"ema_{period}"] = _ema(out["close"], period)
    out = out.join(_macd(out["close"], params.macd_fast, params.macd_slow, params.macd_signal))
    out = out.join(_kdj(out, params.kdj_period, params.kdj_smooth))
    out = out.join(_dmi(out, params.dmi_period))
    out["atr"] = _atr(out, params.atr_period)
    return out"""

    code_indicator_run = """indicator_frames = {symbol: compute_indicator_panel(df) for symbol, df in raw_data.items()}
print(f"完成指标计算: {len(indicator_frames)} 只标的")
display(indicator_frames[sample_symbol].head())"""

    markdown_signals = """## 信号规则
按照 draft 中“t 日上穿 + t-2 / t-3 日仍位于下方”的定义生成 16 个信号，并实现趋势/阈值组合。"""

    code_signals = """SIGNAL_ORDER = [
    "KDJ",
    "KDJ-with-trend",
    "KDJ-with-limit",
    "KDJ-with-limit-trend",
    "MACD",
    "MACD-with-trend",
    "MACD-with-limit",
    "MACD-with-trend-limit",
    "EMA5-10",
    "EMA5-10-with-trend",
    "EMA5-20",
    "EMA5-20-with-trend",
    "ADX-ADXR",
    "ADX-ADXR-with-trend",
    "PDI-MDI",
    "PDI-MDI-with-trend",
]


def cross_over(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    cond = (
        (series_a > series_b)
        & (series_a.shift(2) < series_b.shift(2))
        & (series_a.shift(3) < series_b.shift(3))
    )
    return cond.fillna(False)


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    trend = df["ema_5"] > df["ema_60"]

    kdj_cross = cross_over(df["kdj_k"], df["kdj_d"])
    macd_cross = cross_over(df["macd_dif"], df["macd_dea"])
    ema5_10_cross = cross_over(df["ema_5"], df["ema_10"])
    ema5_20_cross = cross_over(df["ema_5"], df["ema_20"])
    adx_adxr_cross = cross_over(df["adx"], df["adxr"])
    pdi_mdi_cross = cross_over(df["pdi"], df["mdi"])

    signal_df = pd.DataFrame(index=df.index)
    signal_df["KDJ"] = kdj_cross
    signal_df["KDJ-with-trend"] = kdj_cross & trend
    signal_df["KDJ-with-limit"] = kdj_cross & (df["kdj_k"] < 50)
    signal_df["KDJ-with-limit-trend"] = signal_df["KDJ-with-limit"] & trend

    signal_df["MACD"] = macd_cross
    signal_df["MACD-with-trend"] = macd_cross & trend
    signal_df["MACD-with-limit"] = macd_cross & (df["macd_dif"] < 0)
    signal_df["MACD-with-trend-limit"] = signal_df["MACD-with-limit"] & trend

    signal_df["EMA5-10"] = ema5_10_cross
    signal_df["EMA5-10-with-trend"] = ema5_10_cross & trend
    signal_df["EMA5-20"] = ema5_20_cross
    signal_df["EMA5-20-with-trend"] = ema5_20_cross & trend

    signal_df["ADX-ADXR"] = adx_adxr_cross
    signal_df["ADX-ADXR-with-trend"] = adx_adxr_cross & trend
    signal_df["PDI-MDI"] = pdi_mdi_cross
    signal_df["PDI-MDI-with-trend"] = pdi_mdi_cross & trend

    return signal_df.astype(bool)"""

    code_signal_run = """signal_frames = {symbol: build_signals(df) for symbol, df in indicator_frames.items()}
full_frames = {
    symbol: indicator_frames[symbol].join(signal_frames[symbol], how="left")
    for symbol in indicator_frames
}
print(f"联合指标 + 信号数据准备完成，示例列: {list(full_frames[sample_symbol].columns)[:8]}")"""

    markdown_triple = """## 三围栏标签计算
对每个信号触发事件，在 horizon ∈ [5,15]（步长 2）与 ATR 倍数 ∈ [1.0, 3.0]（步长 0.2）的网格上计算三围栏标签：
- 第一触发止盈（high ≥ 止盈价）记为 1
- 第一触发止损（low ≤ 止损价）记为 -1
- 在持有期内未触发则记为 0
若同日同时命中止盈/止损，优先视作止盈。"""

    code_generate_events = """def generate_events(
    frames: Dict[str, pd.DataFrame],
    signal_names: Iterable[str],
    horizons: List[int],
    atr_multipliers: List[float],
    reward_ratio: float,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Tuple[int, str, str, pd.Timestamp, pd.Timestamp, int, float, int]] = []
    event_index: Dict[Tuple[str, str, pd.Timestamp], int] = {}
    next_event_id = 0
    max_horizon = max(horizons)

    for symbol, df in frames.items():
        close = df["close"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        atr = df["atr"].to_numpy()
        index = df.index.to_list()

        for signal in signal_names:
            if signal not in df:
                continue
            signal_values = df[signal].fillna(False).to_numpy(dtype=bool)
            trigger_positions = np.where(signal_values)[0]
            if trigger_positions.size == 0:
                continue

            for pos in trigger_positions:
                entry_date = index[pos]
                if not (analysis_start <= entry_date <= analysis_end):
                    continue
                atr_value = atr[pos]
                if np.isnan(atr_value) or atr_value <= 0:
                    continue
                if pos + max_horizon >= len(df):
                    continue

                key = (signal, symbol, entry_date)
                if key not in event_index:
                    event_index[key] = next_event_id
                    next_event_id += 1
                event_id = event_index[key]

                entry_price = close[pos]
                future_highs = high[pos + 1 : pos + 1 + max_horizon]
                future_lows = low[pos + 1 : pos + 1 + max_horizon]
                future_dates = index[pos + 1 : pos + 1 + max_horizon]

                for horizon in horizons:
                    sub_highs = future_highs[:horizon]
                    sub_lows = future_lows[:horizon]
                    sub_dates = future_dates[:horizon]

                    for atr_mult in atr_multipliers:
                        stop_loss = entry_price - atr_value * atr_mult
                        take_profit = entry_price + atr_value * atr_mult * reward_ratio

                        hit_stop = np.where(sub_lows <= stop_loss)[0]
                        hit_take = np.where(sub_highs >= take_profit)[0]
                        first_stop = int(hit_stop[0]) if hit_stop.size else None
                        first_take = int(hit_take[0]) if hit_take.size else None

                        if first_stop is None and first_take is None:
                            label = 0
                            exit_date = sub_dates[-1]
                        elif first_take is None:
                            label = -1
                            exit_date = sub_dates[first_stop]
                        elif first_stop is None:
                            label = 1
                            exit_date = sub_dates[first_take]
                        else:
                            if first_take <= first_stop:
                                label = 1
                                exit_date = sub_dates[first_take]
                            else:
                                label = -1
                                exit_date = sub_dates[first_stop]

                        records.append(
                            (
                                event_id,
                                signal,
                                symbol,
                                entry_date,
                                exit_date,
                                horizon,
                                float(atr_mult),
                                int(label),
                            )
                        )

    events_df = pd.DataFrame(
        records,
        columns=[
            "event_id",
            "signal",
            "symbol",
            "entry_date",
            "exit_date",
            "horizon",
            "atr_mult",
            "label",
        ],
    )
    base_counts = pd.Series(event_index).reset_index(name="event_id")
    base_counts = base_counts.groupby("level_0").size().reset_index(name="unique_triggers")
    base_counts = base_counts.rename(columns={"level_0": "signal"})
    return events_df, base_counts"""

    code_generate_run = """%%time
events_df, base_trigger_counts = generate_events(
    frames=full_frames,
    signal_names=SIGNAL_ORDER,
    horizons=HORIZONS,
    atr_multipliers=ATR_MULTIPLIERS,
    reward_ratio=REWARD_RATIO,
    analysis_start=ANALYSIS_START,
    analysis_end=ANALYSIS_END,
)
print(events_df.shape)
display(events_df.head())"""

    markdown_metrics = """## 指标统计
基于事件数据统计胜率与净胜次数，分别寻找胜率 / 净胜次数最高的组合。"""

    code_metrics = """def count_wins(series: pd.Series) -> int:
    return int((series == 1).sum())


def count_losses(series: pd.Series) -> int:
    return int((series == -1).sum())


metrics_df = (
    events_df.groupby(["signal", "horizon", "atr_mult"])
    .agg(
        trades=("label", "count"),
        wins=("label", count_wins),
        losses=("label", count_losses),
        net=("label", "sum"),
    )
    .reset_index()
)
metrics_df["win_rate"] = metrics_df["wins"] / metrics_df["trades"]

best_win_df = (
    metrics_df.sort_values(["signal", "win_rate", "net"], ascending=[True, False, False])
    .groupby("signal")
    .head(1)
    .reset_index(drop=True)
)

best_net_df = (
    metrics_df.sort_values(["signal", "net", "win_rate"], ascending=[True, False, False])
    .groupby("signal")
    .head(1)
    .reset_index(drop=True)
)

summary_win = best_win_df.merge(base_trigger_counts, on="signal", how="left")
summary_net = best_net_df.merge(base_trigger_counts, on="signal", how="left")

print("最佳胜率组合预览：")
display(summary_win.head())
print("最佳净胜组合预览：")
display(summary_net.head())"""

    markdown_tables = "## 汇总表（Notebook 展示）"

    code_tables = """display(summary_win.sort_values("win_rate", ascending=False).reset_index(drop=True))
display(summary_net.sort_values("net", ascending=False).reset_index(drop=True))"""

    markdown_assets = """## 可视化资产
生成用于 HTML 报告的图像：
- 2024 年信号示例蜡烛图（根据最佳胜率组合选择触发次数最多的标的）
- 胜率热力图 + 每个 horizon 最优 ATR 倍数折线
- 净胜热力图 + 每个 horizon 最优 ATR 倍数折线"""

    code_assets = """ASSET_DIR.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    return name.lower().replace(" ", "-")


def select_demo_symbol(signal: str, horizon: int, atr_mult: float, events: pd.DataFrame) -> str:
    subset = events[
        (events["signal"] == signal)
        & (events["horizon"] == horizon)
        & (events["atr_mult"] == atr_mult)
    ]
    subset_2024 = subset[(subset["entry_date"] >= PLOT_START) & (subset["entry_date"] <= PLOT_END)]
    if not subset_2024.empty:
        counts = subset_2024.groupby("symbol")["event_id"].nunique().sort_values(ascending=False)
    else:
        counts = subset.groupby("symbol")["event_id"].nunique().sort_values(ascending=False)
    if counts.empty:
        return active_symbols[0]
    return counts.index[0]


def plot_signal_demo(
    df: pd.DataFrame,
    events: pd.DataFrame,
    signal: str,
    symbol: str,
    horizon: int,
    atr_mult: float,
    output_path: Path,
) -> None:
    plot_df = df.loc[PLOT_START:PLOT_END]
    if plot_df.empty:
        raise ValueError(f"{symbol} 在 {PLOT_START:%Y-%m-%d}~{PLOT_END:%Y-%m-%d} 无数据")

    mask = (
        (events["signal"] == signal)
        & (events["symbol"] == symbol)
        & (events["horizon"] == horizon)
        & (events["atr_mult"] == atr_mult)
        & (events["entry_date"] >= PLOT_START)
        & (events["entry_date"] <= PLOT_END)
    )
    event_slice = events.loc[mask].sort_values("entry_date")

    positions = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(13, 6))
    candle_width = 0.6
    for idx, (_, row) in zip(positions, plot_df.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        ax.add_patch(Rectangle((idx - candle_width / 2, lower), candle_width, height, facecolor=color, edgecolor=color))

    ema_colors = {
        "ema_5": "#1f77b4",
        "ema_10": "#ff7f0e",
        "ema_20": "#2ca02c",
        "ema_60": "#9467bd",
    }
    for col, color in ema_colors.items():
        ax.plot(positions, plot_df[col].values, label=col.upper(), color=color)

    color_map = {1: "#ef4444", 0: "#facc15", -1: "#22c55e"}
    for _, ev in event_slice.iterrows():
        if ev["entry_date"] not in plot_df.index:
            continue
        start_idx = plot_df.index.get_loc(ev["entry_date"])
        if ev["exit_date"] in plot_df.index:
            end_idx = plot_df.index.get_loc(ev["exit_date"])
        else:
            end_idx = len(plot_df) - 1
        ax.axvspan(start_idx - 0.5, end_idx + 0.5, color=color_map[ev["label"]], alpha=0.18)

    ax.set_title(f"{symbol} — {signal} 示例 ({PLOT_START:%Y})")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")

    tick_step = max(len(plot_df) // 10, 1)
    tick_positions = list(range(0, len(plot_df), tick_step))
    if tick_positions[-1] != len(plot_df) - 1:
        tick_positions.append(len(plot_df) - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([plot_df.index[i].strftime("%Y-%m-%d") for i in tick_positions], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    best_line: pd.Series,
    output_path: Path,
    cmap: str,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(matrix.values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Holding horizon (days)")
    ax.set_ylabel("ATR multiple")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.plot(np.arange(len(best_line)), best_line.values, color="white", marker="o", linewidth=1.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


assets: Dict[str, Dict[str, Path | str | float | int]] = {}

for _, row in summary_win.iterrows():
    signal = row["signal"]
    horizon = int(row["horizon"])
    atr_mult = float(row["atr_mult"])
    slug = slugify(signal)

    demo_symbol = select_demo_symbol(signal, horizon, atr_mult, events_df)
    demo_chart_path = ASSET_DIR / f"{slug}_demo.png"
    plot_signal_demo(full_frames[demo_symbol], events_df, signal, demo_symbol, horizon, atr_mult, demo_chart_path)

    signal_metrics = metrics_df[metrics_df["signal"] == signal]
    win_matrix = signal_metrics.pivot(index="atr_mult", columns="horizon", values="win_rate").sort_index()
    net_matrix = signal_metrics.pivot(index="atr_mult", columns="horizon", values="net").sort_index()

    best_atr_by_h = win_matrix.idxmax()
    win_heatmap_path = ASSET_DIR / f"{slug}_winrate.png"
    plot_heatmap(win_matrix, f"{signal} 胜率", best_atr_by_h, win_heatmap_path, cmap="viridis", colorbar_label="Win rate")

    best_net_by_h = net_matrix.idxmax()
    net_heatmap_path = ASSET_DIR / f"{slug}_net.png"
    plot_heatmap(net_matrix, f"{signal} 净胜次数", best_net_by_h, net_heatmap_path, cmap="RdYlGn", colorbar_label="Net wins")

    assets[signal] = {
        "demo_symbol": demo_symbol,
        "demo_path": demo_chart_path,
        "win_heatmap": win_heatmap_path,
        "net_heatmap": net_heatmap_path,
        "horizon": horizon,
        "atr_mult": atr_mult,
    }

print(f"生成图像 {len(assets)} 套，保存路径: {ASSET_DIR}")"""

    markdown_html = """## HTML 报告导出
将 Notebook 中整理的表格与图片写入 HTML 文档，便于分享。"""

    code_html = """best_win_lookup = summary_win.set_index("signal").to_dict("index")
best_net_lookup = summary_net.set_index("signal").to_dict("index")

summary_win_html = summary_win.sort_values("win_rate", ascending=False).to_html(
    index=False, float_format="{:.2f}".format
)
summary_net_html = summary_net.sort_values("net", ascending=False).to_html(
    index=False, float_format="{:.2f}".format
)

section_html_list: List[str] = []
for signal, info in assets.items():
    win_meta = best_win_lookup.get(signal, {})
    net_meta = best_net_lookup.get(signal, {})
    section_html_list.append(
        f\"\"\"
        <section class="signal-block">
          <h2>{signal}</h2>
          <p>示例标的：{info["demo_symbol"]}；胜率最优组合：horizon={info["horizon"]}，ATR×={info["atr_mult"]:.1f}</p>
          <ul>
            <li>触发次数：{int(win_meta.get("unique_triggers", 0))}</li>
            <li>最佳胜率：{win_meta.get("win_rate", float("nan")):.2%}（净胜 {int(win_meta.get("net", 0))}）</li>
            <li>最佳净胜：h={int(net_meta.get("horizon", info["horizon"]))}，ATR×={float(net_meta.get("atr_mult", info["atr_mult"])):.1f}，净胜 {int(net_meta.get("net", 0))}</li>
          </ul>
          <figure>
            <img src="{info['demo_path'].name}" alt="{signal} demo">
            <figcaption>2024 年示例 + 三围栏标签着色</figcaption>
          </figure>
          <div class="heatmaps">
            <figure>
              <img src="{info['win_heatmap'].name}" alt="{signal} win rate">
              <figcaption>胜率热力图</figcaption>
            </figure>
            <figure>
              <img src="{info['net_heatmap'].name}" alt="{signal} net wins">
              <figcaption>净胜热力图</figcaption>
            </figure>
          </div>
        </section>
        \"\"\"
    )

html_body = "".join(section_html_list)

html = f\"\"\"<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Signal Test Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ color: #2563eb; margin-top: 24px; }}
    table {{ border-collapse: collapse; margin: 16px 0; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 10px; text-align: right; }}
    th {{ background-color: #f3f4f6; }}
    td:first-child, th:first-child {{ text-align: left; }}
    section.signal-block {{ border-top: 1px solid #e5e7eb; padding-top: 16px; margin-top: 24px; }}
    figure {{ margin: 12px 0; }}
    figure img {{ max-width: 100%; height: auto; border: 1px solid #e5e7eb; }}
    .heatmaps {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .heatmaps figure {{ flex: 1 1 260px; }}
  </style>
</head>
<body>
  <h1>Signal Test Report (2020-2025)</h1>
  <p>持有期 {HORIZONS} 天，ATR 倍数 {ATR_MULTIPLIERS}，止盈：止损 = {REWARD_RATIO:.1f}:1。</p>
  <h2>胜率最佳组合</h2>
  {summary_win_html}
  <h2>净胜次数最佳组合</h2>
  {summary_net_html}
  {html_body}
</body>
</html>\"\"\"

HTML_REPORT_PATH.write_text(html, encoding="utf-8")
print(f"HTML 报告已生成: {HTML_REPORT_PATH}")
display(HTML(f'<a href="{HTML_REPORT_PATH}">{HTML_REPORT_PATH}</a>'))"""

    markdown_closing = """## 下一步
- 若需在 Notebook 中继续分析，可直接复用 `events_df` 与 `metrics_df`
- 可根据报告结果挑选感兴趣的组合进一步做回测或交易模拟"""

    return [
        ("markdown", markdown_intro),
        ("markdown", markdown_overview),
        ("code", code_imports),
        ("markdown", markdown_universe),
        ("code", code_parameters),
        ("markdown", markdown_load),
        ("code", code_load_fn),
        ("code", code_load_run),
        ("markdown", markdown_indicators),
        ("code", code_indicators),
        ("code", code_indicator_run),
        ("markdown", markdown_signals),
        ("code", code_signals),
        ("code", code_signal_run),
        ("markdown", markdown_triple),
        ("code", code_generate_events),
        ("code", code_generate_run),
        ("markdown", markdown_metrics),
        ("code", code_metrics),
        ("markdown", markdown_tables),
        ("code", code_tables),
        ("markdown", markdown_assets),
        ("code", code_assets),
        ("markdown", markdown_html),
        ("code", code_html),
        ("markdown", markdown_closing),
    ]


def main() -> None:
    output_path = Path("module-experiments/signal-test/signal_analysis.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    for cell_type, content in build_cells():
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content))
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

    nbf.write(nb, output_path)
    print(f"Notebook written to {output_path}")


if __name__ == "__main__":
    main()
