"""计算传统技术指标并绘制示例行情图。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_US
from qlib.data import D

DATA_DIR = Path(__file__).resolve().parent.parent / "data/qlib_us_selected/qlib_data"


@dataclass
class IndicatorParams:
    ema_periods: Tuple[int, ...] = (5, 10, 20, 60)
    boll_period: int = 20
    boll_std: float = 2.0
    sar_step: float = 0.02
    sar_max: float = 0.2
    kc_period: int = 20
    kc_atr_mult: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    dmi_period: int = 14
    dma_short: int = 10
    dma_long: int = 50
    dma_signal: int = 10
    adtm_period: int = 23
    cci_period: int = 14
    kdj_period: int = 9
    kdj_smooth: int = 3
    rsi_period: int = 14
    arbr_period: int = 26
    psy_period: int = 12
    vr_period: int = 26
    ddi_period: int = 13
    dpo_period: int = 20
    osc_period: int = 20
    osc_signal_period: int = 6
    atr_period: int = 14
    mi_ema_period: int = 9
    mi_sum_period: int = 25
    srmi_period: int = 14


class TechnicalIndicatorCalculator:
    """封装传统技术指标的计算逻辑，统一输出完整指标表。"""

    def __init__(self, data: pd.DataFrame, params: IndicatorParams | None = None) -> None:
        self.data = data.sort_index().copy()
        self.params = params or IndicatorParams()

    def compute(self) -> pd.DataFrame:
        df = self.data.copy()
        p = self.params

        for period in p.ema_periods:
            ema_col = f"ema_{period}"
            df[ema_col] = self._ema(df["close"], period)
            df[f"{ema_col}_slope"] = df[ema_col] - df[ema_col].shift(1)

        df[["boll_mid", "boll_up", "boll_low"]] = self._bollinger(df["close"], p.boll_period, p.boll_std)
        df["sar"] = self._sar(df, p.sar_step, p.sar_max)
        df[["kc_mid", "kc_up", "kc_low"]] = self._keltner(df, p.kc_period, p.kc_atr_mult)

        macd = self._macd(df["close"], p.macd_fast, p.macd_slow, p.macd_signal)
        df = df.join(macd)

        dmi = self._dmi(df, p.dmi_period)
        df = df.join(dmi)

        df[["dma", "ama"]] = self._dma(df["close"], p.dma_short, p.dma_long, p.dma_signal)
        df[["adtm", "adtmma"]] = self._adtm(df, p.adtm_period)
        df["cci"] = self._cci(df, p.cci_period)
        df[["kdj_k", "kdj_d", "kdj_j"]] = self._kdj(df, p.kdj_period, p.kdj_smooth)
        df["rsi"] = self._rsi(df["close"], p.rsi_period)
        df[["ar", "br"]] = self._arbr(df, p.arbr_period)
        df["psy"] = self._psy(df["close"], p.psy_period)
        df["vr"] = self._vr(df, p.vr_period)
        df["atr"] = self._atr(df, p.atr_period)
        df[["ddi", "ddi_dmz", "ddi_dmf"]] = self._ddi(df, p.ddi_period)
        df["dpo"] = self._dpo(df["close"], p.dpo_period)
        df[["osc", "osc_signal"]] = self._osc(df["close"], p.osc_period, p.osc_signal_period)
        df["mi"] = self._mass_index(df, p.mi_ema_period, p.mi_sum_period)
        df["srmi"] = self._srmi(df["close"], p.srmi_period)

        return df

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _bollinger(series: pd.Series, period: int, nbdev: float) -> pd.DataFrame:
        mid = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        up = mid + nbdev * std
        low = mid - nbdev * std
        return pd.DataFrame({"boll_mid": mid, "boll_up": up, "boll_low": low})

    @staticmethod
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

    def _sar(self, df: pd.DataFrame, step: float, max_step: float) -> pd.Series:
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

    def _keltner(self, df: pd.DataFrame, period: int, mult: float) -> pd.DataFrame:
        mid = self._ema(df["close"], period)
        atr = self._true_range(df).rolling(window=period).mean()
        up = mid + mult * atr
        low = mid - mult * atr
        return pd.DataFrame({"kc_mid": mid, "kc_up": up, "kc_low": low})

    def _macd(self, series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
        ema_fast = self._ema(series, fast)
        ema_slow = self._ema(series, slow)
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd = (dif - dea) * 2
        return pd.DataFrame({"macd_dif": dif, "macd_dea": dea, "macd_hist": macd})

    def _dmi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        tr = self._true_range(df)
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

    def _dma(self, series: pd.Series, short: int, long: int, signal: int) -> pd.DataFrame:
        diff = self._ema(series, short) - self._ema(series, long)
        ama = diff.rolling(window=signal).mean()
        return pd.DataFrame({"dma": diff, "ama": ama})

    def _adtm(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        open_prev = df["open"].shift(1)
        dtm = np.where(df["open"] > open_prev, np.maximum(df["high"] - df["open"], df["open"] - open_prev), 0.0)
        dbm = np.where(df["open"] < open_prev, np.maximum(df["open"] - df["low"], open_prev - df["open"]), 0.0)
        stm = pd.Series(dtm, index=df.index).rolling(window=period).sum()
        sbm = pd.Series(dbm, index=df.index).rolling(window=period).sum()
        adtm = (stm - sbm) / np.where(stm > sbm, stm, sbm)
        adtm = adtm.replace([np.inf, -np.inf], np.nan)
        adtmma = adtm.rolling(window=period).mean()
        return pd.DataFrame({"adtm": adtm, "adtmma": adtmma})

    def _cci(self, df: pd.DataFrame, period: int) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        ma = tp.rolling(window=period).mean()
        md = (tp - ma).abs().rolling(window=period).mean()
        return (tp - ma) / (0.015 * md)

    def _kdj(self, df: pd.DataFrame, period: int, smooth: int) -> pd.DataFrame:
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        rsv = (df["close"] - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(alpha=1 / smooth, adjust=False).mean()
        d = k.ewm(alpha=1 / smooth, adjust=False).mean()
        j = 3 * k - 2 * d
        return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _arbr(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        ar_numerator = (df["high"] - df["open"]).rolling(window=period).sum()
        ar_denominator = (df["open"] - df["low"]).rolling(window=period).sum()
        ar = ar_numerator * 100 / ar_denominator

        br_numerator = (df["high"] - df["close"].shift(1)).rolling(window=period).sum()
        br_denominator = (df["close"].shift(1) - df["low"]).rolling(window=period).sum()
        br = br_numerator * 100 / br_denominator
        return pd.DataFrame({"ar": ar, "br": br})

    def _psy(self, series: pd.Series, period: int) -> pd.Series:
        up_days = (series.diff() > 0).astype(int)
        return up_days.rolling(window=period).sum() * 100 / period

    def _vr(self, df: pd.DataFrame, period: int) -> pd.Series:
        close_diff = df["close"].diff()
        vol = df["volume"]
        av = vol.where(close_diff > 0, 0)
        bv = vol.where(close_diff < 0, 0)
        cv = vol.where(close_diff == 0, 0)
        av_sum = av.rolling(window=period).sum()
        bv_sum = bv.rolling(window=period).sum()
        cv_sum = cv.rolling(window=period).sum()
        return (av_sum + 0.5 * cv_sum) * 100 / (bv_sum + 0.5 * cv_sum)

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        tr = self._true_range(df)
        return tr.rolling(window=period).mean()

    def _ddi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        high_diff = (high - high.shift(1)).abs()
        low_diff = (low - low.shift(1)).abs()
        dmz = np.where((high_diff > low_diff) & (high > high.shift(1)), high_diff, 0.0)
        dmf = np.where((low_diff > high_diff) & (low < low.shift(1)), low_diff, 0.0)
        dmz_series = pd.Series(dmz, index=df.index)
        dmf_series = pd.Series(dmf, index=df.index)
        dmz_sum = dmz_series.rolling(window=period).sum()
        dmf_sum = dmf_series.rolling(window=period).sum()
        total = dmz_sum + dmf_sum
        ddi = (dmz_sum - dmf_sum) / total.replace(0, np.nan)
        return pd.DataFrame({
            "ddi": ddi,
            "ddi_dmz": dmz_sum,
            "ddi_dmf": dmf_sum,
        })

    def _dpo(self, series: pd.Series, period: int) -> pd.Series:
        shift = int(period / 2) + 1
        sma = series.rolling(window=period).mean()
        return series - sma.shift(shift)

    def _osc(self, series: pd.Series, period: int, signal_period: int) -> pd.DataFrame:
        sma = series.rolling(window=period).mean()
        osc = series - sma
        osc_signal = osc.ewm(span=signal_period, adjust=False).mean()
        return pd.DataFrame({"osc": osc, "osc_signal": osc_signal})

    def _mass_index(self, df: pd.DataFrame, ema_period: int, sum_period: int) -> pd.Series:
        hl_range = df["high"] - df["low"]
        ema1 = hl_range.ewm(span=ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=ema_period, adjust=False).mean()
        mass = (ema1 / ema2).rolling(window=sum_period).sum()
        return mass

    def _srmi(self, series: pd.Series, period: int) -> pd.Series:
        lowest = series.rolling(window=period).min()
        highest = series.rolling(window=period).max()
        denominator = (highest - lowest).replace(0, np.nan)
        srmi = (series - lowest) * 100 / denominator
        return srmi


def fetch_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """通过 qlib 接口获取指定区间的日级行情。"""
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    raw = D.features([symbol], fields=fields, start_time=start, end_time=end, freq="day")
    if raw.empty:
        raise ValueError(f"No data returned for {symbol} between {start} and {end}")
    df = raw.xs(symbol, level="instrument")
    df = df.rename(columns={name: name.lstrip("$") for name in df.columns})
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    else:
        df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    return df


def plot_kline_with_indicators(df: pd.DataFrame, ema_columns: Iterable[str]) -> None:
    """在示例图中叠加 EMA、MACD 与 KDJ，便于快速检视指标。"""
    required_cols = list(ema_columns) + ["macd_hist", "macd_dif", "macd_dea", "kdj_k", "kdj_d", "kdj_j"]
    df_plot = df.dropna(subset=required_cols)
    if df_plot.empty:
        raise ValueError("Insufficient data to plot indicators; check input date range and rolling windows.")

    positions = np.arange(len(df_plot))

    fig, (ax_price, ax_macd, ax_kdj) = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    candle_width = 0.6
    for idx, (_, row) in zip(positions, df_plot.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
        ax_price.add_patch(candle)

    for ema_col in ema_columns:
        ax_price.plot(positions, df_plot[ema_col], label=ema_col.upper())

    ax_price.set_title("TSM Daily OHLC with EMA")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)
    ax_price.legend(loc="upper left")

    ax_macd.bar(
        positions,
        df_plot["macd_hist"],
        color=["red" if val >= 0 else "green" for val in df_plot["macd_hist"]],
        alpha=0.5,
    )
    ax_macd.plot(positions, df_plot["macd_dif"], label="DIF", color="blue")
    ax_macd.plot(positions, df_plot["macd_dea"], label="DEA", color="orange")
    ax_macd.set_ylabel("MACD")
    ax_macd.grid(True, linestyle="--", alpha=0.3)
    ax_macd.legend(loc="upper left")

    ax_kdj.plot(positions, df_plot["kdj_k"], label="K", color="blue")
    ax_kdj.plot(positions, df_plot["kdj_d"], label="D", color="orange")
    ax_kdj.plot(positions, df_plot["kdj_j"], label="J", color="green")
    ax_kdj.set_ylabel("KDJ")
    ax_kdj.grid(True, linestyle="--", alpha=0.3)
    ax_kdj.legend(loc="upper left")

    tick_step = max(len(df_plot) // 10, 1)
    tick_positions = list(range(0, len(df_plot), tick_step))
    if tick_positions[-1] != len(df_plot) - 1:
        tick_positions.append(len(df_plot) - 1)
    tick_labels = [df_plot.index[min(pos, len(df_plot) - 1)].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_kdj.set_xticks(tick_positions)
    ax_kdj.set_xticklabels(tick_labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """演示指标计算与可视化流程。"""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"QLib data directory not found: {DATA_DIR}. Run build_us_dataset.py to create it."
        )

    qlib.init(provider_uri=str(DATA_DIR), region=REG_US)

    symbol = "TSM"
    start = "2025-01-01"
    end = "2025-09-30"

    df = fetch_price_data(symbol, start, end)

    calculator = TechnicalIndicatorCalculator(df)
    indicator_df = calculator.compute()

    plot_kline_with_indicators(indicator_df, [f"ema_{p}" for p in calculator.params.ema_periods])


if __name__ == "__main__":
    main()
