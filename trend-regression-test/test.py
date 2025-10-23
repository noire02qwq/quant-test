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

from data_pipeline import apply_scaler, build_dataloaders, load_scaler, prepare_feature_dataframe, split_by_date
from models import create_model
from train import evaluate, EXPERIMENT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trend regression model")
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
            preds = model(inputs)
            batch = preds.size(0)
            for i in range(batch):
                ts = dataset.timestamps[offset + i]
                symbol = dataset.symbols[offset + i]
                records.append(
                    {
                        "date": pd.Timestamp(ts),
                        "symbol": symbol,
                        "true_value": float(targets[i].cpu().item()),
                        "pred_value": float(preds[i].cpu().item()),
                    }
                )
            offset += batch
    return pd.DataFrame(records)


def plot_kline_with_regression(
    price_df: pd.DataFrame,
    values: pd.Series,
    symbol: str,
    title: str,
    output_path: Path,
) -> None:
    if price_df.empty or values.empty:
        raise ValueError("缺少绘图所需的数据。")

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    if isinstance(values, pd.DataFrame):
        if "trend_value" not in values.columns:
            raise KeyError("DataFrame must contain 'trend_value' column for regression plot.")
        value_df = values[["date", "trend_value"]].copy()
    else:
        value_df = values.to_frame(name="trend_value").copy()
    value_df["date"] = pd.to_datetime(value_df["date"])
    if "symbol" not in value_df.columns:
        value_df["symbol"] = symbol
    value_df = value_df.drop_duplicates("date")

    df = df.merge(value_df, on=["date", "symbol"], how="left")

    positions = np.arange(len(df))
    fig, (ax_price, ax_value) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})

    candle_width = 0.6
    for idx, row in df.iterrows():
        color = "red" if row["close"] >= row["open"] else "green"
        ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
        ax_price.add_patch(candle)

    ax_price.grid(True, linestyle="--", alpha=0.3)
    ax_price.set_ylabel("Price")
    ax_price.set_title(title)

    ax_value.plot(positions, df["trend_value"], color="#1f77b4", label="Trend t-value")
    ax_value.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
    ax_value.set_ylabel("t-value")
    ax_value.grid(True, linestyle="--", alpha=0.3)
    ax_value.legend(loc="upper left")

    tick_step = max(len(df) // 12, 1)
    tick_positions = list(range(0, len(df), tick_step))
    if tick_positions[-1] != len(df) - 1:
        tick_positions.append(len(df) - 1)
    tick_labels = [df.loc[pos, "date"].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_value.set_xticks(tick_positions)
    ax_value.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_json(Path(args.config))
    run_dir = Path(args.run_dir or config.get("output_dir", EXPERIMENT_ROOT / "outputs" / "training"))
    metadata_path = run_dir / f"metadata_{config.get('model', {}).get('name', 'transformer')}_{config.get('feature_mode', 'raw_alpha158')}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到 metadata.json，请确认训练流程已完成：{metadata_path}")
    metadata = load_json(metadata_path)

    feature_mode = metadata["feature_mode"]
    feature_cols = metadata["feature_cols"]
    window_size = int(metadata["window_size"])
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
    model = create_model(model_name, input_dim=len(feature_cols), seq_len=window_size, **model_params)

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
        mse, mae = evaluate(model, loader, device)
        print(f"[{split_name}] mse={mse:.4f} mae={mae:.4f}")

        if split_name == "test":
            pred_df = gather_predictions(model, loader, device)
            if pred_df.empty:
                continue
            split_frame = frames[split_name].copy()
            if split_frame.empty:
                continue

            preds_path = run_dir / "test_predictions.csv"
            pred_df.to_csv(preds_path, index=False)
            print(f"[Export] Stored test predictions: {preds_path}")

            for symbol, price_df in split_frame.groupby("symbol"):
                price_df = price_df.sort_values("date")
                values = pred_df[pred_df["symbol"] == symbol][["date", "true_value", "pred_value"]]
                if values.empty:
                    continue
                true_series = values.rename(columns={"true_value": "trend_value"})[["date", "trend_value"]]
                pred_series = values.rename(columns={"pred_value": "trend_value"})[["date", "trend_value"]]
                plot_kline_with_regression(
                    price_df,
                    true_series,
                    symbol,
                    f"{symbol} Test Period - True t-value",
                    run_dir / f"{symbol}_test_true_tvalue.png",
                )
                plot_kline_with_regression(
                    price_df,
                    pred_series,
                    symbol,
                    f"{symbol} Test Period - Predicted t-value",
                    run_dir / f"{symbol}_test_pred_tvalue.png",
                )
                print(
                    f"[Plot] {symbol} test charts saved: "
                    f"{symbol}_test_true_tvalue.png, {symbol}_test_pred_tvalue.png"
                )


if __name__ == "__main__":
    main()
