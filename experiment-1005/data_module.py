from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from indicator_pipeline import (
    DATA_ROOT,
    compute_alpha_features,
    compute_long_strategy_labels,
    compute_secondary_signals,
    compute_traditional_indicators,
    load_symbol_frame,
)

FeatureMode = str


@dataclass
class ScalerState:
    mean: pd.Series
    std: pd.Series

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "mean": self.mean.values.astype(np.float32),
            "std": self.std.values.astype(np.float32),
            "columns": self.mean.index.tolist(),
        }


class SlidingWindowDataset(Dataset):
    """Sliding window dataset supporting multiple instruments."""

    def __init__(
        self,
        frames: Dict[str, pd.DataFrame],
        feature_cols: Sequence[str],
        window_size: int,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.window = int(window_size)
        self.symbol_index: Dict[str, List[int]] = {}
        self.samples: List[Tuple[np.ndarray, int, str, pd.Timestamp]] = []
        for symbol, df in frames.items():
            if df.empty:
                continue
            values = df[self.feature_cols].to_numpy(dtype=np.float32)
            labels = df["label"].to_numpy(dtype=np.int64)
            dates = df.index.to_numpy()
            for idx in range(self.window - 1, len(df)):
                window = values[idx - self.window + 1 : idx + 1]
                if not np.isfinite(window).all():
                    continue
                target = int(labels[idx])
                self.samples.append((window, target, symbol, pd.Timestamp(dates[idx])))
        if self.samples:
            feature_stack, label_list, symbols, timestamps = zip(*self.samples)
            self.X = torch.from_numpy(np.stack(feature_stack, axis=0))
            self.y = torch.from_numpy(np.array(label_list, dtype=np.int64))
            self.meta = list(zip(symbols, timestamps))
            for idx, (sym, _) in enumerate(self.meta):
                self.symbol_index.setdefault(sym, []).append(idx)
        else:
            self.X = torch.empty((0, self.window, len(self.feature_cols)), dtype=torch.float32)
            self.y = torch.empty((0,), dtype=torch.int64)
            self.meta: List[Tuple[str, pd.Timestamp]] = []
            self.symbol_index = {}

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.X[idx], self.y[idx]

    def sample_meta(self, idx: int) -> Tuple[str, pd.Timestamp]:
        return self.meta[idx]

    def indices_for_symbols(self, symbols: Sequence[str]) -> List[int]:
        out: List[int] = []
        for sym in symbols:
            out.extend(self.symbol_index.get(sym, []))
        return out


def _default_label_cfg(label_cfg: Dict[str, float | int | None] | None) -> Dict[str, float | int]:
    base = {"window_days": 15, "stop_loss_mult": 2.0, "stop_gain_mult": 3.0}
    if not label_cfg:
        return base
    return {**base, **{k: v for k, v in label_cfg.items() if v is not None}}


def _resolve_symbol_list(symbol_spec, root: Path) -> List[str]:
    raw_files = {p.stem.upper() for p in root.glob("*.csv") if p.is_file()}
    if isinstance(symbol_spec, str):
        tag = symbol_spec.strip().lower()
        if tag in {"all", "sp500", "s&p500", "s&p-500"}:
            return sorted(raw_files)
        # Comma-separated manual list
        items = [tok.strip() for tok in symbol_spec.split(",") if tok.strip()]
        if items:
            return [sym.upper() for sym in items]
    if isinstance(symbol_spec, Iterable):
        return [str(sym).upper() for sym in symbol_spec if str(sym).strip()]
    if raw_files:
        return sorted(raw_files)
    # Final fallback: empty list
    return []


def _prepare_symbol_features(
    symbol: str,
    feature_mode: FeatureMode,
    root: Path,
    label_cfg: Dict[str, float | int],
) -> pd.DataFrame:
    raw = load_symbol_frame(symbol, root=root).copy()
    raw[["open", "high", "low", "close", "volume"]] = raw[["open", "high", "low", "close", "volume"]].astype(float)
    raw["vwap"] = (raw["high"] + raw["low"] + raw["close"]) / 3.0

    traditional = compute_traditional_indicators(raw)
    if feature_mode == "raw_traditional":
        feature_df = traditional
    elif feature_mode == "raw_secondary":
        feature_df = compute_secondary_signals(traditional)
    elif feature_mode in {"raw_alpha158", "raw_alpha360"}:
        alpha_kind = "alpha158" if feature_mode.endswith("158") else "alpha360"
        alpha_source = raw[["open", "high", "low", "close", "volume", "vwap"]].copy()
        alpha_features = compute_alpha_features(alpha_source, alpha_kind)
        base_cols = [col for col in ["open", "high", "low", "close", "volume", "vwap", "ret"] if col in traditional.columns]
        base = traditional[base_cols].copy()
        feature_df = pd.concat([base, alpha_features], axis=1)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    # Convert bool columns to float for downstream scaling
    for col in feature_df.columns:
        if feature_df[col].dtype == bool:
            feature_df[col] = feature_df[col].astype(float)

    labels = compute_long_strategy_labels(
        traditional,
        window_days=int(label_cfg["window_days"]),
        stop_loss_mult=float(label_cfg["stop_loss_mult"]),
        stop_gain_mult=float(label_cfg["stop_gain_mult"]),
    )
    feature_df = feature_df.join(labels.rename("label"), how="inner")
    feature_df = feature_df.dropna(axis=0, how="any")
    feature_df = feature_df.sort_index()
    return feature_df


def _split_frames(
    frames: Dict[str, pd.DataFrame],
    splits: Dict[str, Tuple[str, str]],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    result: Dict[str, Dict[str, pd.DataFrame]] = {key: {} for key in splits}

    def _clip(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()

    for symbol, df in frames.items():
        for split_name, (start, end) in splits.items():
            result[split_name][symbol] = _clip(df, start, end)
    return result


def _compute_scaler(train_frames: Dict[str, pd.DataFrame], feature_cols: Sequence[str]) -> ScalerState:
    stacked = pd.concat([df[feature_cols] for df in train_frames.values() if not df.empty], axis=0)
    if stacked.empty:
        raise ValueError("Training data is empty after feature preparation; please adjust date ranges or symbols.")
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=0)
    std.replace(0.0, 1.0, inplace=True)
    return ScalerState(mean=mean, std=std)


def _apply_scaler(frames: Dict[str, pd.DataFrame], feature_cols: Sequence[str], scaler: ScalerState) -> Dict[str, pd.DataFrame]:
    scaled: Dict[str, pd.DataFrame] = {}
    for symbol, df in frames.items():
        if df.empty:
            scaled[symbol] = df
            continue
        features = ((df[feature_cols] - scaler.mean) / scaler.std).astype(np.float32)
        scaled_df = features
        scaled_df["label"] = df["label"].astype(np.int64)
        scaled_df.index = df.index
        scaled[symbol] = scaled_df
    return scaled


def prepare_datasets(
    data_cfg: Dict,
    scaler_override: ScalerState | None = None,
    feature_cols_override: Sequence[str] | None = None,
    quiet: bool = False,
) -> Tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset, List[str], ScalerState, List[str]]:
    feature_mode = str(data_cfg["feature_set"]).lower()
    root = Path(data_cfg.get("raw_data_dir", DATA_ROOT)).resolve()
    symbols = _resolve_symbol_list(data_cfg.get("symbols"), root)
    if not symbols:
        raise ValueError("No symbols resolved for dataset. Please check raw data directory or configuration.")
    shuffle_symbols = data_cfg.get("shuffle_symbols", True)
    symbols = list(symbols)
    if shuffle_symbols:
        shuffle_seed = data_cfg.get("shuffle_seed")
        rng = random.Random(shuffle_seed) if shuffle_seed is not None else random
        rng.shuffle(symbols)
    label_cfg = _default_label_cfg(data_cfg.get("label"))

    frames: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = _prepare_symbol_features(symbol, feature_mode, root, label_cfg)
        frames[symbol] = frame

    if feature_cols_override is not None:
        feature_cols = list(feature_cols_override)
    else:
        feature_cols = sorted({col for df in frames.values() for col in df.columns if col != "label"})
    for symbol, df in frames.items():
        frames[symbol] = df[feature_cols + ["label"]]

    splits_cfg = data_cfg["splits"]
    splits = {
        "train": tuple(splits_cfg["train"]),
        "val": tuple(splits_cfg["val"]),
        "test": tuple(splits_cfg["test"]),
    }
    split_frames = _split_frames(frames, splits)

    scaler = scaler_override or _compute_scaler(split_frames["train"], feature_cols)
    train_scaled = _apply_scaler(split_frames["train"], feature_cols, scaler)
    val_scaled = _apply_scaler(split_frames["val"], feature_cols, scaler)
    test_scaled = _apply_scaler(split_frames["test"], feature_cols, scaler)

    window_size = int(data_cfg["window_size"])
    ds_train = SlidingWindowDataset(train_scaled, feature_cols, window_size)
    ds_val = SlidingWindowDataset(val_scaled, feature_cols, window_size)
    ds_test = SlidingWindowDataset(test_scaled, feature_cols, window_size)
    train_symbol_count = len({sym for sym, _ in ds_train.meta})
    val_symbol_count = len({sym for sym, _ in ds_val.meta})
    test_symbol_count = len({sym for sym, _ in ds_test.meta})
    if not quiet:
        print(
            "[experiment-1005] Dataset summary -> "
            f"train: {len(ds_train)} windows / {train_symbol_count} symbols; "
            f"val: {len(ds_val)} windows / {val_symbol_count} symbols; "
            f"test: {len(ds_test)} windows / {test_symbol_count} symbols."
        )
    return ds_train, ds_val, ds_test, feature_cols, scaler, symbols


def resolve_symbols(
    symbol_spec,
    raw_data_dir: str | Path | None = None,
    min_start: pd.Timestamp | None = None,
    min_end: pd.Timestamp | None = None,
) -> List[str]:
    root = Path(raw_data_dir or DATA_ROOT).resolve()
    symbols = _resolve_symbol_list(symbol_spec, root)
    if not symbols:
        return symbols

    filtered: List[str] = []
    dropped_late_listing: List[str] = []
    dropped_delisted: List[str] = []

    for sym in symbols:
        path = root / f"{sym}.csv"
        try:
            df_dates = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
        except Exception as err:
            print(f"[experiment-1005] Warning: failed to read {path}: {err}")
            continue
        if df_dates.empty:
            print(f"[experiment-1005] Warning: no data rows for {sym}, skipping.")
            continue
        first_date = df_dates["date"].min()
        last_date = df_dates["date"].max()
        if min_start is not None and first_date > min_start:
            dropped_late_listing.append(sym)
            continue
        if min_end is not None and last_date < min_end:
            dropped_delisted.append(sym)
            continue
        filtered.append(sym)

    print(
        "[experiment-1005] Symbol filtering -> "
        f"kept {len(filtered)} / {len(symbols)}; "
        f"late listings dropped: {len(dropped_late_listing)}; "
        f"early delistings dropped: {len(dropped_delisted)}"
    )
    return filtered


def create_dataloaders(
    ds_train: SlidingWindowDataset,
    ds_val: SlidingWindowDataset,
    ds_test: SlidingWindowDataset,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
