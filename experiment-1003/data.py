"""数据加载与特征/标签构建模块（独立实验用）。

本模块仅依赖本目录下的 CSV 数据文件，避免修改仓库其他位置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =============================
# 工具函数：特征工程与标签生成
# =============================

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
    """基于 OHLCV 计算完整一级技术指标与 gap_days（休市天数）。

    指标覆盖：
    - EMA(5/10/20/60) 及斜率
    - BOLL(20, 2)、SAR、Keltner Channel(20)
    - MACD、DMI、DMA、ADTM
    - CCI、KDJ、RSI(14)
    - AR、BR、PSY(12)、VR(26)
    - ATR(14)、Mass Index、SRMI
    - DDI、DPO、OSC(20,6)
    - gap_days
    """

    out = df.copy()
    out["ret"] = out["close"].pct_change()

    # EMA 与斜率
    for p in (5, 10, 20, 60):
        col = f"ema_{p}"
        out[col] = _ema(out["close"], p)
        out[f"{col}_slope"] = out[col].diff()

    # BOLL
    mid = out["close"].rolling(window=20).mean()
    std = out["close"].rolling(window=20).std()
    out["boll_mid"], out["boll_up"], out["boll_low"] = mid, mid + 2 * std, mid - 2 * std

    # SAR
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

    # Keltner Channel(20)
    atr20 = _true_range(out).rolling(window=20).mean()
    kc_mid = _ema(out["close"], 20)
    out["kc_mid"], out["kc_up"], out["kc_low"] = kc_mid, kc_mid + 2 * atr20, kc_mid - 2 * atr20

    # MACD
    ema_fast = _ema(out["close"], 12)
    ema_slow = _ema(out["close"], 26)
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=9, adjust=False).mean()
    out["macd_dif"], out["macd_dea"], out["macd_hist"] = dif, dea, (dif - dea) * 2

    # DMI(14)
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

    # DMA(10,50,signal 10)
    diff = _ema(out["close"], 10) - _ema(out["close"], 50)
    out["dma"], out["ama"] = diff, diff.rolling(window=10).mean()

    # ADTM(23)
    open_prev = out["open"].shift(1)
    dtm = np.where(out["open"] > open_prev, np.maximum(out["high"] - out["open"], out["open"] - open_prev), 0.0)
    dbm = np.where(out["open"] < open_prev, np.maximum(out["open"] - out["low"], open_prev - out["open"]), 0.0)
    stm = pd.Series(dtm, index=out.index).rolling(window=23).sum()
    sbm = pd.Series(dbm, index=out.index).rolling(window=23).sum()
    adtm = (stm - sbm) / np.where(stm > sbm, stm, sbm)
    out["adtm"], out["adtmma"] = pd.Series(adtm).replace([np.inf, -np.inf], np.nan), pd.Series(adtm).rolling(window=23).mean()

    # CCI(14)
    tp = (out["high"] + out["low"] + out["close"]) / 3
    ma = tp.rolling(window=14).mean()
    md = (tp - ma).abs().rolling(window=14).mean()
    denom_cci = (0.015 * md).replace(0, np.nan)
    out["cci"] = (tp - ma) / denom_cci

    # KDJ(9, smooth 3)
    low_min = out["low"].rolling(window=9).min()
    high_max = out["high"].rolling(window=9).max()
    rsv = (out["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    out["kdj_k"], out["kdj_d"], out["kdj_j"] = k, d, j

    # RSI(14)
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["rsi"] = 100 - (100 / (1 + rs))

    # ARBR(26)
    ar_num = (out["high"] - out["open"]).rolling(window=26).sum()
    ar_den = (out["open"] - out["low"]).rolling(window=26).sum().replace(0, np.nan)
    out["ar"] = ar_num * 100 / ar_den
    br_num = (out["high"] - out["close"].shift(1)).rolling(window=26).sum()
    br_den = (out["close"].shift(1) - out["low"]).rolling(window=26).sum().replace(0, np.nan)
    out["br"] = br_num * 100 / br_den

    # PSY(12)
    out["psy"] = (out["close"].diff() > 0).rolling(window=12).sum() * 100 / 12

    # VR(26)
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

    # ATR(14)
    out["atr"] = tr.rolling(window=14).mean()

    # Mass Index(9,25)
    hl = out["high"] - out["low"]
    ema1 = hl.ewm(span=9, adjust=False).mean()
    ema2 = ema1.ewm(span=9, adjust=False).mean()
    out["mi"] = (ema1 / ema2).rolling(window=25).sum()

    # SRMI(14)
    lowest = out["close"].rolling(window=14).min()
    highest = out["close"].rolling(window=14).max()
    denom_srmi = (highest - lowest).replace(0, np.nan)
    out["srmi"] = (out["close"] - lowest) * 100 / denom_srmi

    # DDI(13)
    high_diff = (out["high"] - out["high"].shift(1)).abs()
    low_diff = (out["low"] - out["low"].shift(1)).abs()
    dmz = np.where((high_diff > low_diff) & (out["high"] > out["high"].shift(1)), high_diff, 0.0)
    dmf = np.where((low_diff > high_diff) & (out["low"] < out["low"].shift(1)), low_diff, 0.0)
    dmz_s = pd.Series(dmz, index=out.index).rolling(window=13).sum()
    dmf_s = pd.Series(dmf, index=out.index).rolling(window=13).sum()
    out["ddi_dmz"], out["ddi_dmf"] = dmz_s, dmf_s
    ddi_den = (dmz_s + dmf_s).replace(0, np.nan)
    out["ddi"] = (dmz_s - dmf_s) / ddi_den

    # DPO(20)
    period = 20
    shift = int(period / 2) + 1
    sma = out["close"].rolling(window=period).mean()
    out["dpo"] = out["close"] - sma.shift(shift)

    # OSC(20,6)
    osc = out["close"] - out["close"].rolling(window=20).mean()
    out["osc"], out["osc_signal"] = osc, osc.ewm(span=6, adjust=False).mean()

    # gap_days（休市天数）
    out["gap_days"] = out.index.to_series().diff().dt.days.sub(1).clip(lower=0)

    # 清理无穷/缺失值
    out = out.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return out


def _evaluate_trade(
    df: pd.DataFrame,
    start_idx: int,
    entry_price: float,
    atr_value: float,
    window_days: int,
    stop_loss_mult: float,
    stop_gain_mult: float,
) -> float | None:
    """评估从 start_idx 起的交易结果（1/0/-1）。"""

    if np.isnan(entry_price) or np.isnan(atr_value):
        return None
    entry_date = df.index[start_idx]
    horizon_end = entry_date + pd.Timedelta(days=window_days)
    future_end = df.index.searchsorted(horizon_end, side="right") - 1
    if future_end < start_idx:
        return None

    stop_loss = entry_price - stop_loss_mult * atr_value
    stop_gain = entry_price + stop_gain_mult * atr_value
    outcome = 0.0
    for j in range(start_idx, min(future_end + 1, len(df))):
        row = df.iloc[j]
        if row["low"] <= stop_loss:
            return -1.0
        if row["high"] >= stop_gain:
            return 1.0
    return outcome


def compute_labels_long(df: pd.DataFrame, window_days: int = 15) -> pd.Series:
    """长线标签：窗口 15 日；入场考虑 x 收盘与 x+1 开盘（取最差）。"""

    labels = pd.Series(np.nan, index=df.index, dtype=float)
    total = len(df)
    for i in range(total):
        outcomes: List[float] = []
        # x 收盘 → 次日开始观察
        if i + 1 < total:
            r = _evaluate_trade(
                df,
                start_idx=i + 1,
                entry_price=float(df.iloc[i]["close"]),
                atr_value=float(df.iloc[i]["atr"]),
                window_days=window_days,
                stop_loss_mult=2.0,
                stop_gain_mult=3.0,
            )
            if r is not None:
                outcomes.append(r)
        # x+1 开盘 → 当日即可生效
        if i + 1 < total:
            r = _evaluate_trade(
                df,
                start_idx=i + 1,
                entry_price=float(df.iloc[i + 1]["open"]),
                atr_value=float(df.iloc[i + 1]["atr"]),
                window_days=window_days,
                stop_loss_mult=2.0,  # 与长线一致，用同一规则
                stop_gain_mult=3.0,
            )
            if r is not None:
                outcomes.append(r)
        if outcomes:
            labels.iloc[i] = min(outcomes)
    return labels


# =============================
# 数据集封装
# =============================


@dataclass
class SeriesConfig:
    window_size: int
    feature_cols: List[str]


class WindowDataset(Dataset):
    """滑动窗口数据集，将定长窗口映射为样本，标签为窗口末日的分类值。"""

    def __init__(self, df: pd.DataFrame, cfg: SeriesConfig):
        self.df = df
        self.cfg = cfg
        self.X, self.y, self.index = self._build()

    def _build(self) -> Tuple[torch.Tensor, torch.Tensor, List[pd.Timestamp]]:
        W = self.cfg.window_size
        feats = self.df[self.cfg.feature_cols].values.astype(np.float32)
        labels = self.df["label"].values
        idxs = self.df.index.to_list()

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        i_list: List[pd.Timestamp] = []
        for i in range(W - 1, len(self.df)):
            if np.isnan(labels[i]):
                continue
            window = feats[i - (W - 1) : i + 1]
            if not np.isfinite(window).all():
                continue
            X_list.append(window)
            y_list.append(int(labels[i]))  # 直接使用标签值（本实验二分类 0/1）
            i_list.append(idxs[i])
        X = torch.from_numpy(np.stack(X_list, axis=0))
        y = torch.from_numpy(np.array(y_list, dtype=np.int64))
        return X, y, i_list

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.X[idx], self.y[idx]


def load_dataframe(csv_path: Path, index_col: str | None = "datetime") -> pd.DataFrame:
    """加载 CSV（包含 datetime 索引列），并标准化列名为小写。"""
    df = pd.read_csv(csv_path)
    if index_col is None:
        # 若未提供索引列，尝试自动识别第一列为日期
        idx_name = df.columns[0]
    else:
        idx_name = index_col
    df[idx_name] = pd.to_datetime(df[idx_name])
    df = df.set_index(idx_name).sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def prepare_splits(df: pd.DataFrame, cfg: Dict, window_size: int) -> Tuple[WindowDataset, WindowDataset, WindowDataset, List[str]]:
    """构建特征、标签并切分 train/val/test 数据集。"""

    feat_df = compute_features(df)
    labels = compute_labels_long(feat_df, window_days=int(cfg["label_window_days"]))
    # 将多分类标签（-1/0/1）转换为二分类：胜=1，其余（0或-1）=0
    label_bin = (labels == 1).astype(int)
    feat_df = feat_df.assign(label=label_bin)

    # 按日期切分
    def _clip(df_in: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df_in.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()

    train_df = _clip(feat_df, cfg["splits"]["train"][0], cfg["splits"]["train"][1])
    val_df = _clip(feat_df, cfg["splits"]["val"][0], cfg["splits"]["val"][1])
    test_df = _clip(feat_df, cfg["splits"]["test"][0], cfg["splits"]["test"][1])

    feature_cols = [
        # 原始
        "open", "high", "low", "close", "volume", "ret",
        # EMA 与斜率
        "ema_5", "ema_10", "ema_20", "ema_60",
        "ema_5_slope", "ema_10_slope", "ema_20_slope", "ema_60_slope",
        # BOLL
        "boll_mid", "boll_up", "boll_low",
        # SAR、KC
        "sar", "kc_mid", "kc_up", "kc_low",
        # MACD
        "macd_dif", "macd_dea", "macd_hist",
        # DMI、DMA
        "pdi", "mdi", "adx", "adxr", "dma", "ama",
        # ADTM、CCI、KDJ、RSI
        "adtm", "adtmma", "cci", "kdj_k", "kdj_d", "kdj_j", "rsi",
        # ARBR、PSY、VR
        "ar", "br", "psy", "vr",
        # ATR、Mass Index、SRMI
        "atr", "mi", "srmi",
        # DDI、DPO、OSC
        "ddi", "ddi_dmz", "ddi_dmf", "dpo", "osc", "osc_signal",
        # gap_days
        "gap_days",
    ]

    series_cfg = SeriesConfig(window_size=window_size, feature_cols=feature_cols)
    ds_train = WindowDataset(train_df, series_cfg)
    ds_val = WindowDataset(val_df, series_cfg)
    ds_test = WindowDataset(test_df, series_cfg)
    return ds_train, ds_val, ds_test, feature_cols


# ============== 标准化工具（仅在独立实验内部使用） ==============

def compute_scaler(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, np.ndarray]:
    """基于训练集计算特征均值/标准差。"""
    feats = df[feature_cols].astype(np.float32)
    mean = feats.mean(axis=0).values
    std = feats.std(axis=0, ddof=0).values
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std, "cols": feature_cols}


def apply_scaler(df: pd.DataFrame, scaler: Dict[str, np.ndarray]) -> pd.DataFrame:
    """对指定列进行 z-score 标准化。"""
    cols = scaler["cols"]
    mean = scaler["mean"]
    std = scaler["std"]
    X = df[cols].astype(np.float32).values
    X = (X - mean) / std
    out = df.copy()
    out[cols] = X
    return out


def prepare_frames(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """返回切分后的 DataFrame（含特征与标签），便于外部先做标准化再构建 Dataset。"""
    feat_df = compute_features(df)
    labels = compute_labels_long(feat_df, window_days=int(cfg["label_window_days"]))
    # 根据配置切换二分类或三分类标签
    mode = str(cfg.get("label_mode", "binary")).lower()
    if mode == "binary":
        # 胜=1，非胜(0/-1)=0；保持为浮点，留给 WindowDataset 过滤 NaN
        label_idx = (labels == 1).astype(float)
    else:
        # 多分类：{-1,0,1}->{0,1,2}，保持为浮点，留给 WindowDataset 过滤 NaN
        label_idx = labels.map({-1.0: 0.0, 0.0: 1.0, 1.0: 2.0})
    feat_df = feat_df.assign(label=label_idx)

    def _clip(df_in: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df_in.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()

    train_df = _clip(feat_df, cfg["splits"]["train"][0], cfg["splits"]["train"][1])
    val_df = _clip(feat_df, cfg["splits"]["val"][0], cfg["splits"]["val"][1])
    test_df = _clip(feat_df, cfg["splits"]["test"][0], cfg["splits"]["test"][1])

    feature_cols = [
        "open", "high", "low", "close", "volume", "ret",
        "ema_5", "ema_10", "ema_20", "ema_60",
        "ema_5_slope", "ema_10_slope", "ema_20_slope", "ema_60_slope",
        "boll_mid", "boll_up", "boll_low",
        "sar", "kc_mid", "kc_up", "kc_low",
        "macd_dif", "macd_dea", "macd_hist",
        "pdi", "mdi", "adx", "adxr", "dma", "ama",
        "adtm", "adtmma", "cci", "kdj_k", "kdj_d", "kdj_j", "rsi",
        "ar", "br", "psy", "vr",
        "atr", "mi", "srmi",
        "ddi", "ddi_dmz", "ddi_dmf", "dpo", "osc", "osc_signal",
        "gap_days",
    ]
    return train_df, val_df, test_df, feature_cols


def make_loaders(ds_train: Dataset, ds_val: Dataset, ds_test: Dataset, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def _plot_kline_with_subplots(df: pd.DataFrame, out_path: Path) -> None:
    """绘制日K+EMA（主图），成交量、MACD、KDJ 子图；横轴使用交易日序号连续显示。"""
    if df.empty:
        raise ValueError("No data to plot")
    positions = np.arange(len(df))
    fig, (ax_price, ax_vol, ax_macd, ax_kdj) = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                                                            gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    # 主图：K线 + EMA
    candle_width = 0.6
    for idx, (_, row) in zip(positions, df.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = float(min(row["open"], row["close"]))
        height = float(max(row["open"], row["close"])) - lower or 1e-10
        ax_price.add_patch(Rectangle((idx - candle_width / 2, lower), candle_width, height,
                                     edgecolor=color, facecolor=color))
    for ema_col in ["ema_5", "ema_10", "ema_20", "ema_60"]:
        if ema_col in df.columns:
            ax_price.plot(positions, df[ema_col], label=ema_col.upper())
    ax_price.set_title("TSM 2024 Daily K with EMA")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)
    ax_price.legend(loc="upper left")

    # 成交量
    colors = ["red" if c >= o else "green" for o, c in zip(df["open"], df["close"])]
    ax_vol.bar(positions, df["volume"], color=colors, alpha=0.6)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(True, linestyle="--", alpha=0.3)

    # MACD
    if {"macd_hist", "macd_dif", "macd_dea"}.issubset(df.columns):
        ax_macd.bar(positions, df["macd_hist"], color=["red" if v >= 0 else "green" for v in df["macd_hist"]], alpha=0.5)
        ax_macd.plot(positions, df["macd_dif"], label="DIF", color="blue")
        ax_macd.plot(positions, df["macd_dea"], label="DEA", color="orange")
        ax_macd.set_ylabel("MACD")
        ax_macd.grid(True, linestyle="--", alpha=0.3)
        ax_macd.legend(loc="upper left")

    # KDJ
    if {"kdj_k", "kdj_d", "kdj_j"}.issubset(df.columns):
        ax_kdj.plot(positions, df["kdj_k"], label="K", color="blue")
        ax_kdj.plot(positions, df["kdj_d"], label="D", color="orange")
        ax_kdj.plot(positions, df["kdj_j"], label="J", color="green")
        ax_kdj.set_ylabel("KDJ")
        ax_kdj.grid(True, linestyle="--", alpha=0.3)
        ax_kdj.legend(loc="upper left")

    # X 轴刻度
    tick_step = max(len(df) // 10, 1)
    tick_positions = list(range(0, len(df), tick_step))
    if tick_positions[-1] != len(df) - 1:
        tick_positions.append(len(df) - 1)
    tick_labels = [df.index[min(pos, len(df) - 1)].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_kdj.set_xticks(tick_positions)
    ax_kdj.set_xticklabels(tick_labels, rotation=45, ha="right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """数据模块主函数：
    1) 加载扩展 CSV 并计算完整一级指标；
    2) 截取 2024 年区间，绘制 K/EMA + 成交量 + MACD + KDJ；
    3) 保存至 experiment-1003/outputs/plots/tsm_2024_overview.png。
    """
    from json import loads

    cfg = loads(Path("experiment-1003/config.json").read_text(encoding="utf-8"))
    df_raw = load_dataframe(Path(cfg["data_path"]), index_col=cfg.get("index_col", "datetime"))
    df_feat = compute_features(df_raw)

    # 截取 2024 年绘图
    df_2024 = df_feat.loc[pd.Timestamp("2024-01-01"): pd.Timestamp("2024-12-31")].copy()
    _plot_kline_with_subplots(df_2024, Path("experiment-1003/outputs/plots/tsm_2024_overview.png"))


if __name__ == "__main__":
    main()
