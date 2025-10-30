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
- 汇总每个信号的胜率、相对胜率与净胜次数
- 导出单张 Excel 工作表，汇总所有 (信号, horizon, ATR 倍数) 的统计结果"""

    code_imports = """from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from IPython.display import display"""

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
EXCEL_PATH = OUTPUT_DIR / "signal_report.xlsx"

ANALYSIS_START = pd.Timestamp("2020-01-01")
ANALYSIS_END = pd.Timestamp("2025-08-31")
BUFFER_DAYS = 60
HORIZONS = list(range(5, 16, 2))
ATR_MULTIPLIERS = [round(x, 1) for x in np.linspace(1.0, 3.0, num=11)]
REWARD_RATIO = 1.5

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
按照 draft 中“t 日上穿 + t-2 / t-3 日仍位于下方”的定义生成 18 个信号，其中 KDJ-with-limit 被拆分为 50/40/30 三个阈值，以便对比限制强度。"""

    code_signals = """SIGNAL_ORDER = [
    "KDJ",
    "KDJ-with-trend",
    "KDJ-with-limit-50",
    "KDJ-with-limit-40",
    "KDJ-with-limit-30",
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
    signal_df["KDJ-with-limit-50"] = kdj_cross & (df["kdj_k"] < 50)
    signal_df["KDJ-with-limit-40"] = kdj_cross & (df["kdj_k"] < 40)
    signal_df["KDJ-with-limit-30"] = kdj_cross & (df["kdj_k"] < 30)
    signal_df["KDJ-with-limit-trend"] = signal_df["KDJ-with-limit-50"] & trend

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
    signal_names: Sequence[str],
    horizons: Sequence[int],
    atr_multipliers: Sequence[float],
    reward_ratio: float,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Tuple[int, str, str, pd.Timestamp, pd.Timestamp, int, float, int]] = []
    event_index: Dict[Tuple[str, str, pd.Timestamp], int] = {}
    next_event_id = 0
    max_horizon = max(horizons)
    required_entry_cols = ["close", "high", "low", "atr"]

    for symbol, df in frames.items():
        if df[required_entry_cols].isna().any().any():
            df = df.dropna(subset=required_entry_cols, how="any")
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
                if pos >= len(index):
                    continue
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

                    if len(sub_highs) < horizon or len(sub_lows) < horizon:
                        continue
                    if np.isnan(sub_highs).any() or np.isnan(sub_lows).any():
                        continue

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
denom = metrics_df["wins"] + metrics_df["losses"]
metrics_df["rel_win_rate"] = metrics_df["wins"] / denom
metrics_df.loc[denom == 0, "rel_win_rate"] = np.nan

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

display(summary_win.head())
display(summary_net.head())"""

    markdown_tables = "## 汇总表（Notebook 展示）"

    code_tables = """display(summary_win.sort_values("win_rate", ascending=False).reset_index(drop=True))
display(summary_net.sort_values("net", ascending=False).reset_index(drop=True))"""

    markdown_summary = """## 汇总统计
- 可视化步骤已移除，仅在 Notebook 中展示关键表格
- 统计结果保存为单张 Excel 工作表，包含所有 (信号, 持有期, ATR 倍数) 的胜率与净胜指标"""

    code_notice = """print("可视化已跳过，仅生成表格数据。")"""

    code_excel_helper = """def ensure_openpyxl() -> None:
    try:
        import openpyxl  # type: ignore[import-untyped]
        return
    except ModuleNotFoundError:
        pass

    import sys
    from pathlib import Path

    base_prefix = Path(sys.prefix).parents[1]
    candidates = sorted(base_prefix.glob("lib/python*/site-packages"), reverse=True)
    for candidate in candidates:
        if not candidate.exists():
            continue
        if str(candidate) in sys.path:
            continue
        sys.path.append(str(candidate))
        try:
            import openpyxl  # type: ignore[import-untyped]
            return
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError(
        "openpyxl 未安装，无法导出 Excel。请在当前环境运行 `pip install openpyxl` 后重试。"
    )"""

    code_excel = """ensure_openpyxl()
metrics_df.sort_values(["signal", "horizon", "atr_mult"]).to_excel(EXCEL_PATH, index=False)
print(f"Excel 报告已生成: {EXCEL_PATH}")"""

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
        ("markdown", markdown_summary),
        ("code", code_notice),
        ("code", code_excel_helper),
        ("code", code_excel),
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
