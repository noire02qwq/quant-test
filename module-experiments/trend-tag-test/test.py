from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch

from data_pipeline import (
    TREND_CLASS_MAP,
    apply_scaler,
    build_dataloaders,
    load_scaler,
    prepare_feature_dataframe,
    split_by_date,
)
from models import create_model
from train import evaluate, EXPERIMENT_ROOT


INV_LABEL_MAP = {v: k for k, v in TREND_CLASS_MAP.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model for trend-tag-test experiment")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (defaults to best model)")
    parser.add_argument("--split", default="test", help="Split name or 'all'")
    parser.add_argument("--run-dir", default=None, help="Override run/output directory")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def gather_predictions(model: torch.nn.Module, loader, device: torch.device) -> pd.DataFrame:
    dataset = loader.dataset
    records = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            batch = preds.size(0)
            for i in range(batch):
                ts = dataset.timestamps[offset + i]
                symbol = dataset.symbols[offset + i]
                records.append(
                    {
                        "date": pd.Timestamp(ts),
                        "symbol": symbol,
                        "true_label": int(targets[i].cpu().item()),
                        "pred_label": int(preds[i].cpu().item()),
                    }
                )
            offset += batch
    return pd.DataFrame(records)


def _highlight_classes(ax: plt.Axes, classes: pd.Series) -> None:
    values = classes.fillna(0).to_numpy(dtype=int)
    color_map = {1: "#86efac", -1: "#fca5a5"}
    start = None
    current = None
    for idx, val in enumerate(values):
        if val == 0:
            if start is not None and current in color_map:
                ax.axvspan(start - 0.5, idx - 0.5, color=color_map[current], alpha=0.2)
                start = None
                current = None
            continue
        if start is None:
            start = idx
            current = val
        elif val != current:
            if current in color_map:
                ax.axvspan(start - 0.5, idx - 0.5, color=color_map[current], alpha=0.2)
            start = idx
            current = val
    if start is not None and current in color_map:
        ax.axvspan(start - 0.5, len(values) - 0.5, color=color_map[current], alpha=0.2)


def plot_kline_with_background(
    price_df: pd.DataFrame,
    class_df: pd.DataFrame,
    symbol: str,
    title: str,
    output_path: Path,
) -> None:
    if price_df.empty or class_df.empty:
        raise ValueError("缺少绘图所需的数据。")

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "trend_class" in df.columns:
        df = df.drop(columns=["trend_class"])
    df = df.sort_values("date").reset_index(drop=True)
    if "symbol" not in df.columns:
        df["symbol"] = symbol
    class_df = class_df.copy()
    if "symbol" not in class_df.columns:
        class_df["symbol"] = symbol
    df = df.merge(class_df, on=["date", "symbol"], how="left")

    positions = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(16, 6))
    candle_width = 0.6
    for idx, row in df.iterrows():
        color = "red" if row["close"] >= row["open"] else "green"
        ax.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
        ax.add_patch(candle)

    _highlight_classes(ax, df["trend_class"])
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.3)

    tick_step = max(len(df) // 12, 1)
    tick_positions = list(range(0, len(df), tick_step))
    if tick_positions[-1] != len(df) - 1:
        tick_positions.append(len(df) - 1)
    tick_labels = [df.loc[pos, "date"].strftime("%Y-%m-%d") for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_json(Path(args.config))
    run_dir = Path(args.run_dir or config.get("output_dir", EXPERIMENT_ROOT / "outputs" / "training"))
    metadata_path = run_dir / f"metadata_{config.get('model', {}).get('name', 'transformer')}_{config.get('feature_mode', 'raw_traditional')}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到 metadata.json，请确认训练流程已完成：{metadata_path}")
    metadata = load_json(metadata_path)

    feature_mode = metadata["feature_mode"]
    feature_cols = metadata["feature_cols"]
    window_size = int(metadata["window_size"])
    num_classes = int(metadata.get("num_classes", 3))
    splits = metadata.get("splits", config.get("splits", {}))

    frame = prepare_feature_dataframe(feature_mode)
    frames = split_by_date(frame, splits)

    scaler = load_scaler(Path(metadata["scaler_path"]))
    scaled_frames = {name: apply_scaler(df, scaler) for name, df in frames.items()}

    loaders = build_dataloaders(
        scaled_frames,
        feature_cols,
        window_size,
        batch_size=int(metadata.get("batch_size", config.get("batch_size", 128))),
        num_workers=int(config.get("num_workers", 0)),
    )

    model_name = metadata["model_name"]
    model_params = metadata.get("model_params", {})
    model = create_model(model_name, input_dim=len(feature_cols), seq_len=window_size, num_classes=num_classes, **model_params)

    checkpoint_path = Path(args.checkpoint or metadata.get("best_checkpoint", ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"指定或默认的 checkpoint 不存在：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model.to(device)

    target_splits: Sequence[str]
    if args.split.lower() == "all":
        target_splits = list(loaders.keys())
    else:
        if args.split not in loaders:
            raise KeyError(f"未知数据切分：{args.split}，可选 {list(loaders.keys())}")
        target_splits = [args.split]

    for split_name in target_splits:
        loader = loaders[split_name]
        loss, acc = evaluate(model, loader, device)
        print(f"[{split_name}] loss={loss:.4f} acc={acc:.4f}")

        if split_name == "test":
            pred_df = gather_predictions(model, loader, device)
            if pred_df.empty:
                continue
            pred_df["true_class"] = pred_df["true_label"].map(INV_LABEL_MAP)
            pred_df["pred_class"] = pred_df["pred_label"].map(INV_LABEL_MAP)

            split_frame = frames[split_name].copy()
            if split_frame.empty:
                continue

            for symbol, price_df in split_frame.groupby("symbol"):
                price_df = price_df.sort_values("date")
                real_series = pred_df[pred_df["symbol"] == symbol][["date", "true_class"]]
                pred_series = pred_df[pred_df["symbol"] == symbol][["date", "pred_class"]]
                if real_series.empty or pred_series.empty:
                    continue
                real_series = (
                    real_series.rename(columns={"true_class": "trend_class"})
                    .drop_duplicates("date")
                    .assign(symbol=symbol)
                )
                pred_series = (
                    pred_series.rename(columns={"pred_class": "trend_class"})
                    .drop_duplicates("date")
                    .assign(symbol=symbol)
                )
                real_series["date"] = pd.to_datetime(real_series["date"])
                pred_series["date"] = pd.to_datetime(pred_series["date"])
                plot_kline_with_background(
                    price_df,
                    real_series,
                    symbol,
                    f"{symbol} Test Period - True Classes",
                    run_dir / f"{symbol}_test_true_classes.png",
                )
                plot_kline_with_background(
                    price_df,
                    pred_series,
                    symbol,
                    f"{symbol} Test Period - Predicted Classes",
                    run_dir / f"{symbol}_test_pred_classes.png",
                )
                print(
                    f"[Plot] Test visualizations saved: {symbol}_test_true_classes.png, "
                    f"{symbol}_test_pred_classes.png"
                )


if __name__ == "__main__":
    main()
