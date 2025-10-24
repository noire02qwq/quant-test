from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from compute_labels import compute_regression_labels


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXPERIMENT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
QLIB_DIR = DATA_ROOT / "qlib"
DEFAULT_SYMBOLS: Tuple[str, ...] = ("TSM",)


def load_raw_frame(symbol: str = "TSM") -> pd.DataFrame:
    """加载原始 CSV，返回按日期排序的 DataFrame。"""

    csv_path = RAW_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到原始数据文件：{csv_path}，请先运行 prepare_data.py")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.columns = [col.lower() for col in df.columns]
    df = df.sort_values("date").set_index("date")
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["ret"] = df["close"].pct_change()
    df["gap_days"] = df.index.to_series().diff().dt.days.sub(1)
    return df


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


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


def compute_traditional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """复刻主流程中的传统技术指标集合。"""

    out = df.copy()

    for period in (5, 10, 20, 60):
        ema_col = f"ema_{period}"
        out[ema_col] = _ema(out["close"], period)
        out[f"{ema_col}_slope"] = out[ema_col] - out[ema_col].shift(1)

    out[["boll_mid", "boll_up", "boll_low"]] = _bollinger(out["close"], period=20, nbdev=2.0)
    out["sar"] = _sar(out, step=0.02, max_step=0.2)
    out[["kc_mid", "kc_up", "kc_low"]] = _keltner(out, period=20, mult=2.0)

    macd = _macd(out["close"], fast=12, slow=26, signal=9)
    out = out.join(macd)

    dmi = _dmi(out, period=14)
    out = out.join(dmi)

    out[["dma", "ama"]] = _dma(out["close"], short=10, long=50, signal=10)
    out[["adtm", "adtmma"]] = _adtm(out, period=23)
    out["cci"] = _cci(out, period=14)
    out[["kdj_k", "kdj_d", "kdj_j"]] = _kdj(out, period=9, smooth=3)
    out["rsi"] = _rsi(out["close"], period=14)
    out[["ar", "br"]] = _arbr(out, period=26)
    out["psy"] = _psy(out["close"], period=12)
    out["vr"] = _vr(out, period=26)
    out["atr"] = _atr(out, period=14)
    out[["ddi", "ddi_dmz", "ddi_dmf"]] = _ddi(out, period=13)
    out["dpo"] = _dpo(out["close"], period=20)
    out[["osc", "osc_signal"]] = _osc(out["close"], period=20, signal_period=6)
    out["mi"] = _mass_index(out, ema_period=9, sum_period=25)
    out["srmi"] = _srmi(out["close"], period=14)

    return out


def _bollinger(series: pd.Series, period: int, nbdev: float) -> pd.DataFrame:
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    up = mid + nbdev * std
    low = mid - nbdev * std
    return pd.DataFrame({"boll_mid": mid, "boll_up": up, "boll_low": low})


def _sar(df: pd.DataFrame, step: float, max_step: float) -> pd.Series:
    high = df["high"].values
    low = df["low"].values
    sar = np.zeros_like(high)
    trend = 1
    af = step
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, len(high)):
        prev_sar = sar[i - 1]
        if trend == 1:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[i])
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = step
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1], high[i])
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = step
    return pd.Series(sar, index=df.index)


def _keltner(df: pd.DataFrame, period: int, mult: float) -> pd.DataFrame:
    mid = _ema(df["close"], period)
    atr = _true_range(df).rolling(window=period).mean()
    up = mid + mult * atr
    low = mid - mult * atr
    return pd.DataFrame({"kc_mid": mid, "kc_up": up, "kc_low": low})


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = (dif - dea) * 2
    return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": macd})


def _dmi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    tr = _true_range(df)
    tr_n = tr.rolling(window=period).sum()
    up_move = df["high"].diff()
    down_move = df["low"].diff() * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = pd.Series(plus_dm, index=df.index).rolling(window=period).sum() * 100 / tr_n
    minus_di = pd.Series(minus_dm, index=df.index).rolling(window=period).sum() * 100 / tr_n
    dx = (plus_di - minus_di).abs() * 100 / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    adxr = (adx + adx.shift(period)) / 2
    return pd.DataFrame({"pdi": plus_di, "mdi": minus_di, "adx": adx, "adxr": adxr})


def _dma(series: pd.Series, short: int, long: int, signal: int) -> pd.DataFrame:
    diff = _ema(series, short) - _ema(series, long)
    ama = diff.rolling(window=signal).mean()
    return pd.DataFrame({"dma": diff, "ama": ama})


def _adtm(df: pd.DataFrame, period: int) -> pd.DataFrame:
    open_prev = df["open"].shift(1)
    dtm = np.where(df["open"] > open_prev, np.maximum(df["high"] - df["open"], df["open"] - open_prev), 0.0)
    dbm = np.where(df["open"] < open_prev, np.maximum(df["open"] - df["low"], open_prev - df["open"]), 0.0)
    stm = pd.Series(dtm, index=df.index).rolling(window=period).sum()
    sbm = pd.Series(dbm, index=df.index).rolling(window=period).sum()
    adtm = (stm - sbm) / np.where(stm > sbm, stm, sbm)
    adtm = pd.Series(adtm, index=df.index).replace([np.inf, -np.inf], np.nan)
    adtmma = adtm.rolling(window=period).mean()
    return pd.DataFrame({"adtm": adtm, "adtmma": adtmma})


def _cci(df: pd.DataFrame, period: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(window=period).mean()
    md = (tp - ma).abs().rolling(window=period).mean()
    denom = (0.015 * md).replace(0, np.nan)
    return (tp - ma) / denom


def _kdj(df: pd.DataFrame, period: int, smooth: int) -> pd.DataFrame:
    low_min = df["low"].rolling(window=period, min_periods=1).min()
    high_max = df["high"].rolling(window=period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    rsv = (df["close"] - low_min) / denom * 100
    k = rsv.ewm(com=smooth - 1, adjust=False).mean()
    d = k.ewm(com=smooth - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window=period).mean()
    down = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _arbr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    ar = ((df["high"] - df["open"]).rolling(window=period).sum()) / ((df["open"] - df["low"]).rolling(window=period).sum())
    br = ((df["high"] - df["close"].shift(1)).abs().rolling(window=period).sum()) / (
        (df["close"].shift(1) - df["low"]).abs().rolling(window=period).sum()
    )
    return pd.DataFrame({"ar": ar, "br": br})


def _psy(series: pd.Series, period: int) -> pd.Series:
    up_days = (series.diff() > 0).astype(float)
    return up_days.rolling(window=period).mean() * 100


def _vr(df: pd.DataFrame, period: int) -> pd.Series:
    close_change = df["close"].diff()
    av = df["volume"].where(close_change > 0, 0.0)
    bv = df["volume"].where(close_change < 0, 0.0)
    cv = df["volume"].where(close_change == 0, 0.0)
    numerator = av.rolling(window=period).sum() + cv.rolling(window=period).sum() / 2
    denominator = bv.rolling(window=period).sum() + cv.rolling(window=period).sum() / 2
    return numerator / denominator


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    tr = _true_range(df)
    return tr.rolling(window=period, min_periods=period).mean()


def _ddi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    dmz = df["high"].diff()
    dmf = df["low"].diff().abs()
    dmz = np.where(dmz > 0, dmz, 0.0)
    dmf = np.where(dmf > dmz, dmf, dmz)
    dmz = pd.Series(dmz, index=df.index).rolling(window=period).sum()
    dmf = pd.Series(dmf, index=df.index).rolling(window=period).sum()
    ddi = (dmz - dmf) / (dmz + dmf)
    return pd.DataFrame({"ddi": ddi, "ddi_dmz": dmz, "ddi_dmf": dmf})


def _dpo(series: pd.Series, period: int) -> pd.Series:
    ma = series.rolling(window=period).mean()
    return series - ma.shift(int(period / 2) + 1)


def _osc(series: pd.Series, period: int, signal_period: int) -> pd.DataFrame:
    osc = series - series.shift(period)
    osc_signal = osc.rolling(window=signal_period).mean()
    return pd.DataFrame({"osc": osc, "osc_signal": osc_signal})


def _mass_index(df: pd.DataFrame, ema_period: int, sum_period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    ema1 = high_low.ewm(span=ema_period, adjust=False).mean()
    ema2 = ema1.ewm(span=ema_period, adjust=False).mean()
    mass = (ema1 / ema2).rolling(window=sum_period).sum()
    return mass


def _srmi(series: pd.Series, period: int) -> pd.Series:
    diff = series.diff()
    pos = diff.clip(lower=0)
    neg = (-diff).clip(lower=0)
    pos_ma = pos.rolling(window=period).mean()
    neg_ma = neg.rolling(window=period).mean()
    denom = (pos_ma + neg_ma).replace(0, np.nan)
    return (pos_ma - neg_ma) / denom


def _cross_over(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    cond = (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))
    return cond.fillna(False)


def compute_secondary_signals(features: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(index=features.index)

    macd_cross = _cross_over(features["macd_dif"], features["macd_dea"]) & (features["macd_hist"] > 0)
    signals["signal_macd"] = macd_cross

    kdj_cross = _cross_over(features["kdj_k"], features["kdj_d"]) & (features["kdj_k"] <= 70)
    signals["signal_kdj"] = kdj_cross

    ema20 = features["ema_20"]
    ema60 = features["ema_60"]
    ema_cross = (
        (ema20 > ema60)
        & (features["close"] > features["open"])
        & (features["close"] >= ema20)
        & (features["open"] <= ema20)
        & (features["close"].shift(1) <= ema20.shift(1))
    )
    signals["signal_ema"] = ema_cross

    sar_prev_above = features["sar"].shift(1) > features[["open", "close"]].shift(1).max(axis=1)
    sar_now_below = features["sar"] <= features[["open", "close"]].min(axis=1)
    signals["signal_sar"] = sar_prev_above & sar_now_below

    pdi_cross = _cross_over(features["pdi"], features["mdi"])
    adx_cross = _cross_over(features["adx"], features["adxr"])
    signals["signal_dmi"] = pdi_cross | adx_cross

    adtm_cross = _cross_over(features["adtm"], features["adtmma"]) & (features["adtm"] < 0.5)
    signals["signal_adtm"] = adtm_cross

    ddi_series = features["ddi"]
    signals["signal_ddi"] = (ddi_series.shift(1) < 0) & (ddi_series.shift(-1) > 0)

    dpo_series = features["dpo"]
    signals["signal_dpo"] = (dpo_series.shift(1) < 0) & (dpo_series.shift(-1) > 0)

    osc_series = features["osc"]
    osc_signal = features["osc_signal"]
    signals["signal_osc"] = _cross_over(osc_series, osc_signal)

    srmi_series = features["srmi"]
    signals["signal_srmi"] = (srmi_series.shift(1) < 0) & (srmi_series.shift(-1) > 0)

    signals["signal_boll_close_break"] = (features["close"] > features["boll_up"]).fillna(False)
    signals["signal_boll_open_break"] = (features["open"] < features["boll_low"]).fillna(False)

    return pd.concat([features, signals], axis=1)


_ALPHA_VAR_PATTERN = re.compile(r"\$([A-Za-z0-9_]+)")


def _linreg_stats(values: np.ndarray) -> Tuple[float, float, float]:
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    y = values[mask]
    x = np.arange(len(values))[mask].astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return np.nan, np.nan, np.nan
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    y_fit = slope * x + intercept
    resid = y - y_fit
    ss_res = (resid**2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    last_res = resid[-1] if resid.size else np.nan
    return slope, r2, last_res


def _ensure_series(value: pd.Series | float | int, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    return pd.Series(float(value), index=index)


def _build_expression_environment(df: pd.DataFrame) -> Dict[str, object]:
    index = df.index

    def COL(name: str) -> pd.Series:
        key = name.lower()
        if key not in df.columns:
            raise KeyError(f"Column '{key}' not found for alpha feature computation")
        return df[key]

    def Ref(series: pd.Series, n: int) -> pd.Series:
        return _ensure_series(series, index).shift(int(n))

    def Mean(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)
        if window == 0:
            return ser.expanding(min_periods=1).mean()
        return ser.rolling(window, min_periods=1).mean()

    def Std(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)
        if window == 0:
            return ser.expanding(min_periods=1).std(ddof=0)
        return ser.rolling(window, min_periods=1).std(ddof=0)

    def Sum(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)
        if window == 0:
            return ser.expanding(min_periods=1).sum()
        return ser.rolling(window, min_periods=1).sum()

    def Max(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)
        if window == 0:
            return ser.expanding(min_periods=1).max()
        return ser.rolling(window, min_periods=1).max()

    def Min(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)
        if window == 0:
            return ser.expanding(min_periods=1).min()
        return ser.rolling(window, min_periods=1).min()

    def Quantile(series: pd.Series, window: int, q: float) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)
        if window == 0:
            return ser.expanding(min_periods=1).quantile(q)
        return ser.rolling(window, min_periods=1).quantile(q)

    def Rank(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)

        def _rank(arr: np.ndarray) -> float:
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                return np.nan
            last = valid[-1]
            return float((valid <= last).sum() / valid.size)

        if window == 0:
            return ser.expanding(min_periods=1).apply(lambda x: _rank(x.values), raw=False)
        return ser.rolling(window, min_periods=1).apply(lambda x: _rank(x.values), raw=False)

    def IdxMax(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)

        def _idxmax(arr: np.ndarray) -> float:
            if not np.isfinite(arr).any():
                return np.nan
            return float(np.nanargmax(arr) + 1)

        if window == 0:
            return ser.expanding(min_periods=1).apply(lambda x: _idxmax(x.values), raw=False)
        return ser.rolling(window, min_periods=1).apply(lambda x: _idxmax(x.values), raw=False)

    def IdxMin(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)

        def _idxmin(arr: np.ndarray) -> float:
            if not np.isfinite(arr).any():
                return np.nan
            return float(np.nanargmin(arr) + 1)

        if window == 0:
            return ser.expanding(min_periods=1).apply(lambda x: _idxmin(x.values), raw=False)
        return ser.rolling(window, min_periods=1).apply(lambda x: _idxmin(x.values), raw=False)

    def Slope(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)

        def _slope(arr: np.ndarray) -> float:
            slope, _, _ = _linreg_stats(arr)
            return slope

        if window == 0:
            return ser.expanding(min_periods=2).apply(lambda x: _slope(x.values), raw=False)
        return ser.rolling(window, min_periods=2).apply(lambda x: _slope(x.values), raw=False)

    def Rsquare(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)

        def _r2(arr: np.ndarray) -> float:
            _, r2, _ = _linreg_stats(arr)
            return r2

        if window == 0:
            return ser.expanding(min_periods=2).apply(lambda x: _r2(x.values), raw=False)
        return ser.rolling(window, min_periods=2).apply(lambda x: _r2(x.values), raw=False)

    def Greater(left: pd.Series, right: pd.Series | float | int) -> pd.Series:
        lhs = _ensure_series(left, index).astype(float)
        rhs = _ensure_series(right, index).astype(float)
        return pd.Series(np.maximum(lhs.values, rhs.values), index=index)

    def Less(left: pd.Series, right: pd.Series | float | int) -> pd.Series:
        lhs = _ensure_series(left, index).astype(float)
        rhs = _ensure_series(right, index).astype(float)
        return pd.Series(np.minimum(lhs.values, rhs.values), index=index)

    def Abs(series: pd.Series) -> pd.Series:
        return _ensure_series(series, index).abs()

    def Log(series: pd.Series) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)
        return np.log(ser)

    def Corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
        lhs = _ensure_series(left, index).astype(float)
        rhs = _ensure_series(right, index).astype(float)
        if window == 0:
            return lhs.expanding(min_periods=2).corr(rhs)
        return lhs.rolling(window, min_periods=2).corr(rhs)

    def Resi(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)

        def _res(arr: np.ndarray) -> float:
            _, _, last = _linreg_stats(arr)
            return last

        if window == 0:
            return ser.expanding(min_periods=2).apply(lambda x: _res(x.values), raw=False)
        return ser.rolling(window, min_periods=2).apply(lambda x: _res(x.values), raw=False)

    return {
        "COL": COL,
        "Ref": Ref,
        "Mean": Mean,
        "Std": Std,
        "Sum": Sum,
        "Max": Max,
        "Min": Min,
        "Quantile": Quantile,
        "Rank": Rank,
        "IdxMax": IdxMax,
        "IdxMin": IdxMin,
        "Slope": Slope,
        "Rsquare": Rsquare,
        "Greater": Greater,
        "Less": Less,
        "Abs": Abs,
        "Log": Log,
        "Corr": Corr,
        "Resi": Resi,
    }


def _evaluate_alpha_expression(expr: str, df: pd.DataFrame) -> pd.Series:
    from qlib.contrib.data.loader import Alpha158DL, Alpha360DL  # type: ignore

    py_expr = _ALPHA_VAR_PATTERN.sub(lambda m: f"COL('{m.group(1)}')", expr)
    env = _build_expression_environment(df)
    # 定义常量以兼容 qlib 表达式
    env.update({"math": math, "np": np, "pd": pd})
    result = eval(py_expr, {"__builtins__": {}}, env)  # noqa: S307
    return _ensure_series(result, df.index)


def compute_alpha_features(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    from qlib.contrib.data.loader import Alpha158DL, Alpha360DL  # type: ignore

    mode = kind.lower()
    if mode == "alpha158":
        fields, names = Alpha158DL.get_feature_config()
    elif mode == "alpha360":
        fields, names = Alpha360DL.get_feature_config()
    else:
        raise ValueError(f"Unsupported alpha feature kind: {kind}")

    feature_dict: Dict[str, pd.Series] = {}
    for expr, name in zip(fields, names):
        feature_dict[name.lower()] = _evaluate_alpha_expression(expr, df)
    return pd.DataFrame(feature_dict, index=df.index)


def prepare_feature_dataframe(feature_mode: str, symbols: Sequence[str] | None = None) -> pd.DataFrame:
    """根据指定模式生成多标的特征与回归标签。"""

    symbol_list = list(symbols) if symbols else list(DEFAULT_SYMBOLS)
    frames: List[pd.DataFrame] = []

    for symbol in symbol_list:
        raw = load_raw_frame(symbol=symbol)
        traditional = compute_traditional_indicators(raw)
        labels = compute_regression_labels(traditional).rename("label")

        base_cols = ["open", "high", "low", "close", "volume", "vwap", "ret"]
        base_df = traditional[base_cols].copy()

        mode = feature_mode.lower()
        if mode == "raw_traditional":
            feature_df = traditional.copy()
        elif mode == "raw_secondary":
            with_signals = compute_secondary_signals(traditional)
            signal_cols = [col for col in with_signals.columns if col.startswith("signal_")]
            feature_df = pd.concat([base_df, with_signals[signal_cols]], axis=1)
        elif mode in {"raw_alpha158", "raw_alpha360"}:
            alpha_kind = "alpha158" if mode.endswith("158") else "alpha360"
            alpha_source = traditional[["open", "high", "low", "close", "volume", "vwap", "ret"]].copy()
            alpha_features = compute_alpha_features(alpha_source, alpha_kind)
            feature_df = pd.concat([base_df, alpha_features], axis=1)
        else:
            raise ValueError(f"不支持的特征模式：{feature_mode}")

        for col in feature_df.columns:
            if feature_df[col].dtype == bool:
                feature_df[col] = feature_df[col].astype(float)

        dataset = feature_df.join(labels, how="inner")
        dataset = dataset.dropna(axis=0, how="any")
        dataset = dataset.sort_index()
        if dataset.empty:
            continue
        dataset = dataset.reset_index().rename(columns={"index": "date"})
        dataset["symbol"] = symbol
        frames.append(dataset)

    if not frames:
        raise ValueError("所有标的的数据均为空，请检查数据准备流程。")

    combined = pd.concat(frames, axis=0).sort_values(["symbol", "date"]).reset_index(drop=True)
    return combined


def split_by_date(df: pd.DataFrame, splits: Mapping[str, Sequence[str]]) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for split_name, (start, end) in splits.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if "date" not in df.columns:
            raise KeyError("数据集中缺少 'date' 列，无法按日期切分。")
        mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
        clip = df.loc[mask].copy()
        if not clip.empty:
            sort_keys = [col for col in ["symbol", "date"] if col in clip.columns]
            if sort_keys:
                clip = clip.sort_values(sort_keys)
        result[split_name] = clip
    return result


@dataclass
class ScalerState:
    mean: pd.Series
    std: pd.Series

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "columns": self.mean.index.tolist(),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "ScalerState":
        columns = list(data["columns"])  # type: ignore[index]
        mean = pd.Series(data["mean"], index=columns, dtype=np.float32)  # type: ignore[arg-type]
        std = pd.Series(data["std"], index=columns, dtype=np.float32)  # type: ignore[arg-type]
        return cls(mean=mean, std=std)


def fit_scaler(df: pd.DataFrame, feature_cols: Sequence[str]) -> ScalerState:
    stack = df[feature_cols].astype(float)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=0)
    std.replace(0.0, 1.0, inplace=True)
    return ScalerState(mean=mean, std=std)


def apply_scaler(df: pd.DataFrame, scaler: ScalerState) -> pd.DataFrame:
    features = df[scaler.mean.index]
    scaled = (features - scaler.mean) / scaler.std
    scaled = scaled.astype(np.float32)
    scaled_df = scaled.copy()
    scaled_df["label"] = df["label"].astype(np.float32)
    if "symbol" in df.columns:
        scaled_df["symbol"] = df["symbol"].values
    if "date" in df.columns:
        scaled_df["date"] = df["date"].values
    return scaled_df


class SlidingWindowDataset(Dataset):
    """基于单标的的滑动窗口数据集。"""

    def __init__(self, frame: pd.DataFrame, feature_cols: Sequence[str], window_size: int) -> None:
        self.feature_cols = list(feature_cols)
        self.window = int(window_size)
        frame_local = frame.copy()
        if "symbol" not in frame_local.columns:
            frame_local = frame_local.assign(symbol="UNKNOWN")
        if "date" not in frame_local.columns:
            frame_local = frame_local.assign(date=frame_local.index)

        samples: List[np.ndarray] = []
        targets: List[float] = []
        self.timestamps: List[pd.Timestamp] = []
        self.symbols: List[str] = []

        for symbol, sdf in frame_local.groupby("symbol"):
            sdf = sdf.sort_values("date")
            values = sdf[self.feature_cols].to_numpy(dtype=np.float32)
            labels = sdf["label"].to_numpy(dtype=np.float32)
            dates = sdf["date"].to_numpy()
            for idx in range(self.window - 1, len(sdf)):
                window_slice = values[idx - self.window + 1 : idx + 1]
                if not np.isfinite(window_slice).all():
                    continue
                target = labels[idx]
                if not np.isfinite(target):
                    continue
                samples.append(window_slice)
                targets.append(float(target))
                self.timestamps.append(pd.Timestamp(dates[idx]))
                self.symbols.append(str(symbol))

        if samples:
            self.X = torch.from_numpy(np.stack(samples, axis=0))
            self.y = torch.from_numpy(np.array(targets, dtype=np.float32)).unsqueeze(1)
        else:
            self.X = torch.empty((0, self.window, len(self.feature_cols)), dtype=torch.float32)
            self.y = torch.empty((0, 1), dtype=torch.float32)
            self.timestamps = []
            self.symbols = []

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.X[idx], self.y[idx]


def build_dataloaders(
    frames: Mapping[str, pd.DataFrame],
    feature_cols: Sequence[str],
    window_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    datasets = {split: SlidingWindowDataset(frame, feature_cols, window_size) for split, frame in frames.items()}
    loaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        if split == "train":
            loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        else:
            loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return loaders


def save_scaler(path: Path, scaler: ScalerState) -> None:
    path.write_text(json.dumps(scaler.to_json_dict(), ensure_ascii=False, indent=2))


def load_scaler(path: Path) -> ScalerState:
    data = json.loads(path.read_text())
    return ScalerState.from_json_dict(data)


__all__ = [
    "prepare_feature_dataframe",
    "split_by_date",
    "fit_scaler",
    "apply_scaler",
    "SlidingWindowDataset",
    "build_dataloaders",
    "save_scaler",
    "load_scaler",
    "ScalerState",
    "DEFAULT_SYMBOLS",
]
