from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Optional heavy dependency, used for alpha feature definitions
    from qlib.contrib.data.loader import Alpha158DL, Alpha360DL  # type: ignore
except Exception as _QLIB_IMPORT_ERROR:  # pragma: no cover - best effort guard
    Alpha158DL = None  # type: ignore
    Alpha360DL = None  # type: ignore

DATA_ROOT = Path(__file__).resolve().parent / "data" / "raw"


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["close"].pct_change()

    for p in (5, 10, 20, 60):
        col = f"ema_{p}"
        out[col] = _ema(out["close"], p)
        out[f"{col}_slope"] = out[col].diff()

    mid = out["close"].rolling(window=20).mean()
    std = out["close"].rolling(window=20).std()
    out["boll_mid"], out["boll_up"], out["boll_low"] = mid, mid + 2 * std, mid - 2 * std

    def _sar(df_in: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
        high = df_in["high"].values
        low = df_in["low"].values
        sar = np.zeros_like(high)
        trend = 1
        af = step
        ep = high[0]
        sar[0] = low[0]
        for i in range(1, len(high)):
            psar = sar[i - 1]
            if trend == 1:
                sar[i] = psar + af * (ep - psar)
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
                sar[i] = psar + af * (ep - psar)
                sar[i] = max(sar[i], high[i - 1], high[i])
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
                if high[i] > sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = step
        return pd.Series(sar, index=df_in.index)

    out["sar"] = _sar(out)

    atr20 = _true_range(out).rolling(window=20).mean()
    kc_mid = _ema(out["close"], 20)
    out["kc_mid"], out["kc_up"], out["kc_low"] = kc_mid, kc_mid + 2 * atr20, kc_mid - 2 * atr20

    ema_fast = _ema(out["close"], 12)
    ema_slow = _ema(out["close"], 26)
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=9, adjust=False).mean()
    out["macd_dif"], out["macd_dea"], out["macd_hist"] = dif, dea, (dif - dea) * 2

    tr = _true_range(out)
    tr_n = tr.rolling(window=14).sum()
    up_move = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = pd.Series(plus_dm, index=out.index).rolling(window=14).sum() * 100 / tr_n
    minus_di = pd.Series(minus_dm, index=out.index).rolling(window=14).sum() * 100 / tr_n
    dx = (plus_di - minus_di).abs() * 100 / (plus_di + minus_di)
    adx = dx.rolling(window=14).mean()
    adxr = (adx + adx.shift(14)) / 2
    out["pdi"], out["mdi"], out["adx"], out["adxr"] = plus_di, minus_di, adx, adxr

    diff = _ema(out["close"], 10) - _ema(out["close"], 50)
    out["dma"], out["ama"] = diff, diff.rolling(window=10).mean()

    open_prev = out["open"].shift(1)
    dtm = np.where(out["open"] > open_prev, np.maximum(out["high"] - out["open"], out["open"] - open_prev), 0.0)
    dbm = np.where(out["open"] < open_prev, np.maximum(out["open"] - out["low"], open_prev - out["open"]), 0.0)
    stm = pd.Series(dtm, index=out.index).rolling(window=23).sum()
    sbm = pd.Series(dbm, index=out.index).rolling(window=23).sum()
    adtm = (stm - sbm) / np.where(stm > sbm, stm, sbm)
    out["adtm"], out["adtmma"] = pd.Series(adtm).replace([np.inf, -np.inf], np.nan), pd.Series(adtm).rolling(window=23).mean()

    tp = (out["high"] + out["low"] + out["close"]) / 3
    ma = tp.rolling(window=14).mean()
    md = (tp - ma).abs().rolling(window=14).mean()
    denom_cci = (0.015 * md).replace(0, np.nan)
    out["cci"] = (tp - ma) / denom_cci

    low_min = out["low"].rolling(window=9).min()
    high_max = out["high"].rolling(window=9).max()
    rsv = (out["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    out["kdj_k"], out["kdj_d"], out["kdj_j"] = k, d, j

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["rsi"] = 100 - (100 / (1 + rs))

    ar_num = (out["high"] - out["open"]).rolling(window=26).sum()
    ar_den = (out["open"] - out["low"]).rolling(window=26).sum().replace(0, np.nan)
    out["ar"] = ar_num * 100 / ar_den
    br_num = (out["high"] - out["close"].shift(1)).rolling(window=26).sum()
    br_den = (out["close"].shift(1) - out["low"]).rolling(window=26).sum().replace(0, np.nan)
    out["br"] = br_num * 100 / br_den

    out["psy"] = (out["close"].diff() > 0).rolling(window=12).sum() * 100 / 12

    close_diff = out["close"].diff()
    vol = out["volume"]
    av = vol.where(close_diff > 0, 0)
    bv = vol.where(close_diff < 0, 0)
    cv = vol.where(close_diff == 0, 0)
    av_sum = av.rolling(window=26).sum()
    bv_sum = bv.rolling(window=26).sum()
    cv_sum = cv.rolling(window=26).sum()
    vr_den = (bv_sum + 0.5 * cv_sum).replace(0, np.nan)
    out["vr"] = (av_sum + 0.5 * cv_sum) * 100 / vr_den

    out["atr"] = tr.rolling(window=14).mean()

    hl = out["high"] - out["low"]
    ema1 = hl.ewm(span=9, adjust=False).mean()
    ema2 = ema1.ewm(span=9, adjust=False).mean()
    out["mi"] = (ema1 / ema2).rolling(window=25).sum()

    lowest = out["close"].rolling(window=14).min()
    highest = out["close"].rolling(window=14).max()
    denom_srmi = (highest - lowest).replace(0, np.nan)
    out["srmi"] = (out["close"] - lowest) * 100 / denom_srmi

    high_diff = (out["high"] - out["high"].shift(1)).abs()
    low_diff = (out["low"] - out["low"].shift(1)).abs()
    dmz = np.where((high_diff > low_diff) & (out["high"] > out["high"].shift(1)), high_diff, 0.0)
    dmf = np.where((low_diff > high_diff) & (out["low"] < out["low"].shift(1)), low_diff, 0.0)
    dmz_s = pd.Series(dmz, index=out.index).rolling(window=13).sum()
    dmf_s = pd.Series(dmf, index=out.index).rolling(window=13).sum()
    out["ddi_dmz"], out["ddi_dmf"] = dmz_s, dmf_s
    ddi_den = (dmz_s + dmf_s).replace(0, np.nan)
    out["ddi"] = (dmz_s - dmf_s) / ddi_den

    period = 20
    shift = int(period / 2) + 1
    sma = out["close"].rolling(window=period).mean()
    out["dpo"] = out["close"] - sma.shift(shift)

    osc = out["close"] - out["close"].rolling(window=20).mean()
    out["osc"], out["osc_signal"] = osc, osc.ewm(span=6, adjust=False).mean()

    out["gap_days"] = out.index.to_series().diff().dt.days.sub(1).clip(lower=0)

    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out


def load_symbol_frame(symbol: str, root: Path | None = None) -> pd.DataFrame:
    root = root or DATA_ROOT
    symbol_norm = symbol.upper()
    path = root / f"{symbol_norm}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for {symbol_norm}: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_values("date").set_index("date")
    return df


def build_qlib_frame(symbols: Sequence[str], root: Path | None = None) -> pd.DataFrame:
    root = root or DATA_ROOT
    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        df = load_symbol_frame(symbol, root=root)
        df = df.copy()
        df["instrument"] = symbol
        frames.append(df)
    if not frames:
        raise ValueError("No symbols provided to build qlib frame")
    combined = pd.concat(frames, axis=0)
    combined = combined.reset_index().rename(columns={"date": "datetime"})
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined.set_index(["instrument", "datetime"]).sort_index()
    combined.index = pd.MultiIndex.from_tuples(
        [(inst, pd.Timestamp(ts)) for inst, ts in combined.index.to_numpy()],
        names=["instrument", "datetime"],
    )
    return combined


def compute_traditional_indicators(df: pd.DataFrame, ma_windows: Iterable[int] = (5, 20, 60)) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame缺少列: {missing}")
    features = compute_features(df.copy())
    for window in ma_windows:
        features[f"ma_{window}"] = df["close"].rolling(window).mean()
    return features


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

    return pd.concat([features, signals], axis=1)


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
            valid = arr.copy()
            if not np.isfinite(valid).any():
                return np.nan
            return float(np.nanargmax(valid) + 1)

        if window == 0:
            return ser.expanding(min_periods=1).apply(lambda x: _idxmax(x.values), raw=False)
        return ser.rolling(window, min_periods=1).apply(lambda x: _idxmax(x.values), raw=False)

    def IdxMin(series: pd.Series, window: int) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)

        def _idxmin(arr: np.ndarray) -> float:
            valid = arr.copy()
            if not np.isfinite(valid).any():
                return np.nan
            return float(np.nanargmin(valid) + 1)

        if window == 0:
            return ser.expanding(min_periods=1).apply(lambda x: _idxmin(x.values), raw=False)
        return ser.rolling(window, min_periods=1).apply(lambda x: _idxmin(x.values), raw=False)

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

    def _rolling_linreg(series: pd.Series, window: int, kind: str) -> pd.Series:
        ser = _ensure_series(series, index).astype(float)

        def _apply(arr: np.ndarray) -> float:
            slope, r2, resi = _linreg_stats(arr)
            if kind == "slope":
                return slope
            if kind == "rsquare":
                return r2
            return resi

        if window == 0:
            return ser.expanding(min_periods=2).apply(lambda x: _apply(x.values), raw=False)
        return ser.rolling(window, min_periods=2).apply(lambda x: _apply(x.values), raw=False)

    def Slope(series: pd.Series, window: int) -> pd.Series:
        return _rolling_linreg(series, window, "slope")

    def Rsquare(series: pd.Series, window: int) -> pd.Series:
        return _rolling_linreg(series, window, "rsquare")

    def Resi(series: pd.Series, window: int) -> pd.Series:
        return _rolling_linreg(series, window, "resi")

    env: Dict[str, object] = {
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
        "Greater": Greater,
        "Less": Less,
        "Abs": Abs,
        "Log": Log,
        "Corr": Corr,
        "Slope": Slope,
        "Rsquare": Rsquare,
        "Resi": Resi,
        "np": np,
    }
    return env


_ALPHA_VAR_PATTERN = re.compile(r"\$([a-zA-Z_]+)")


def _evaluate_alpha_expression(expr: str, df: pd.DataFrame) -> pd.Series:
    py_expr = _ALPHA_VAR_PATTERN.sub(lambda m: f"COL('{m.group(1)}')", expr)
    env = _build_expression_environment(df)
    result = eval(py_expr, {"__builtins__": {}}, env)  # noqa: S307
    return _ensure_series(result, df.index)


def compute_alpha_features(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if Alpha158DL is None or Alpha360DL is None:
        raise ImportError(f"qlib is required for alpha feature generation: {_QLIB_IMPORT_ERROR}")

    mode = kind.lower()
    if mode == "alpha158":
        fields, names = Alpha158DL.get_feature_config()
    elif mode == "alpha360":
        fields, names = Alpha360DL.get_feature_config()
    else:  # pragma: no cover - safety
        raise ValueError(f"Unsupported alpha feature kind: {kind}")

    result_cols: Dict[str, pd.Series] = {}
    for expr, name in zip(fields, names):
        result_cols[name.lower()] = _evaluate_alpha_expression(expr, df)
    return pd.DataFrame(result_cols, index=df.index)


def _evaluate_trade(
    df: pd.DataFrame,
    start_idx: int,
    entry_price: float,
    atr_value: float,
    window_days: int,
    stop_loss_mult: float,
    stop_gain_mult: float,
) -> float | None:
    if np.isnan(entry_price) or np.isnan(atr_value):
        return None
    if start_idx >= len(df):
        return None
    entry_date = df.index[start_idx]
    horizon_end = entry_date + pd.Timedelta(days=window_days)
    future_end = df.index.searchsorted(horizon_end, side="right") - 1
    if future_end < start_idx:
        return None

    stop_loss = entry_price - stop_loss_mult * atr_value
    stop_gain = entry_price + stop_gain_mult * atr_value
    for j in range(start_idx, min(future_end + 1, len(df))):
        row = df.iloc[j]
        if row["low"] <= stop_loss:
            return -1.0
        if row["high"] >= stop_gain:
            return 1.0
    return 0.0


def compute_long_strategy_labels(
    features: pd.DataFrame,
    window_days: int = 15,
    stop_loss_mult: float = 2.0,
    stop_gain_mult: float = 3.0,
) -> pd.Series:
    labels = pd.Series(np.nan, index=features.index, dtype=float)
    total = len(features)
    for i in range(total):
        outcomes: List[float] = []
        if i + 1 < total:
            r_close = _evaluate_trade(
                features,
                start_idx=i + 1,
                entry_price=float(features.iloc[i]["close"]),
                atr_value=float(features.iloc[i]["atr"]),
                window_days=window_days,
                stop_loss_mult=stop_loss_mult,
                stop_gain_mult=stop_gain_mult,
            )
            if r_close is not None:
                outcomes.append(r_close)
            r_open = _evaluate_trade(
                features,
                start_idx=i + 1,
                entry_price=float(features.iloc[i + 1]["open"]),
                atr_value=float(features.iloc[i + 1]["atr"]),
                window_days=window_days,
                stop_loss_mult=stop_loss_mult,
                stop_gain_mult=stop_gain_mult,
            )
            if r_open is not None:
                outcomes.append(r_open)
        if outcomes:
            labels.iloc[i] = min(outcomes)
    labels = labels.fillna(0.0)
    return (labels == 1.0).astype(int)


__all__ = [
    "DATA_ROOT",
    "load_symbol_frame",
    "build_qlib_frame",
    "compute_traditional_indicators",
    "compute_secondary_signals",
    "compute_alpha_features",
    "compute_long_strategy_labels",
]
