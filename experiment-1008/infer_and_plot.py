from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch

from data_pipeline import (
    apply_scaler,
    build_dataloaders,
    prepare_feature_dataframe,
    split_by_date,
    load_scaler,
)
from models import create_model
from train import EXPERIMENT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference and visualization on test set using best checkpoint.")
    parser.add_argument("--config", required=True, help="路径：训练时使用的配置 JSON。")
    parser.add_argument("--metadata", default=None, help="路径：训练生成的 metadata JSON，默认自动推断。")
    parser.add_argument("--checkpoint", default=None, help="路径：使用的模型权重，默认 metadata 中 best_checkpoint。")
    parser.add_argument("--symbol", default="TSM", help="绘图使用的标的（默认 TSM）。")
    parser.add_argument("--output", default=None, help="输出图像文件路径，默认存放在 run 目录下。")
    parser.add_argument("--device", default=None, help="推理设备（cpu/cuda），默认为训练同配置。")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def gather_predictions(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            batch_pred = model(batch_x)
            preds.append(batch_pred.cpu())
    if not preds:
        return np.array([], dtype=np.float32)
    return torch.cat(preds, dim=0).squeeze(-1).numpy()


def plot_candlestick_with_predictions(
    price_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    symbol: str,
    output_path: Path,
) -> None:
    if price_df.empty or pred_df.empty:
        raise ValueError(f"{symbol} 缺少价格或预测数据，无法绘图。")

    price_df = price_df.sort_values("date")
    pred_df = pred_df.sort_values("date")

    # 对齐预测日期便于绘制参考线
    pred_map = pred_df.set_index("date")

    fig, (ax_price, ax_label) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]})

    # 绘制 K 线图
    positions = np.arange(len(price_df))
    candle_width = 0.6
    for idx, (_, row) in zip(positions, price_df.iterrows()):
        color = "red" if row["close"] >= row["open"] else "green"
        ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        candle = Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color)
        ax_price.add_patch(candle)

    ax_price.set_title(f"{symbol} Test Period K-line")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)

    # 绘制真实标签与预测
    plot_positions = []
    actual_vals = []
    pred_vals = []
    label_dates: List[str] = []
    for idx, date in enumerate(price_df["date"]):
        if date in pred_map.index:
            plot_positions.append(idx)
            entry = pred_map.loc[date]
            actual_vals.append(entry["label"])
            pred_vals.append(entry["prediction"])
            label_dates.append(date.strftime("%Y-%m-%d"))

    ax_label.plot(plot_positions, actual_vals, label="Actual Label", color="#1f77b4")
    ax_label.plot(plot_positions, pred_vals, label="Predicted", color="#d62728", linestyle="--")
    ax_label.set_ylabel("Label / Prediction")
    ax_label.grid(True, linestyle="--", alpha=0.3)
    ax_label.legend(loc="upper left")

    tick_step = max(len(price_df) // 10, 1)
    tick_positions = list(range(0, len(price_df), tick_step))
    if tick_positions[-1] != len(price_df) - 1:
        tick_positions.append(len(price_df) - 1)
    tick_labels = [price_df.iloc[pos]["date"].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_label.set_xticks(tick_positions)
    ax_label.set_xticklabels(tick_labels, rotation=45, ha="right")

    fig.tight_layout()
    ensure_dir(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)

    run_dir = Path(config.get("output_dir", EXPERIMENT_ROOT / "outputs" / "training"))
    metadata_path = Path(args.metadata) if args.metadata else run_dir / f"metadata_{config.get('model', {}).get('name', 'transformer')}_{config.get('feature_mode', 'raw_traditional')}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到 metadata 文件：{metadata_path}")
    metadata = load_json(metadata_path)

    feature_mode = metadata["feature_mode"]
    symbols: Sequence[str] = metadata.get("symbols", [])
    window_size = int(metadata["window_size"])
    splits = metadata.get("splits", config.get("splits", {}))

    frame = prepare_feature_dataframe(feature_mode, symbols=symbols)
    frames = split_by_date(frame, splits)

    scaler_path = Path(metadata["scaler_path"])
    scaler = load_scaler(scaler_path)
    scaled_frames = {name: apply_scaler(df, scaler) for name, df in frames.items()}
    feature_cols = [
        col
        for col in scaled_frames["train"].columns
        if col not in {"label", "symbol", "date"} and np.issubdtype(scaled_frames["train"][col].dtype, np.number)
    ]

    loaders = build_dataloaders(
        scaled_frames,
        feature_cols,
        window_size=window_size,
        batch_size=int(metadata.get("batch_size", config.get("batch_size", 64))),
        num_workers=int(config.get("num_workers", 0)),
    )
    test_loader = loaders["test"]
    test_dataset = test_loader.dataset

    model_name = metadata["model_name"]
    model_params = metadata.get("model_params", {})
    model = create_model(model_name, input_dim=len(feature_cols), seq_len=window_size, **model_params)

    checkpoint_path = Path(args.checkpoint or metadata.get("best_checkpoint", ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint：{checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model.to(device)

    preds = gather_predictions(model, test_loader, device)
    actual = test_dataset.y.squeeze(-1).numpy()
    meta_symbols = getattr(test_dataset, "symbols", ["UNKNOWN"] * len(actual))
    meta_dates = getattr(test_dataset, "timestamps", [pd.Timestamp(0)] * len(actual))

    pred_records = []
    for sym, date, y_true, y_pred in zip(meta_symbols, meta_dates, actual, preds):
        pred_records.append(
            {
                "symbol": sym,
                "date": pd.Timestamp(date),
                "label": float(y_true),
                "prediction": float(y_pred),
            }
        )
    pred_df = pd.DataFrame(pred_records)
    if pred_df.empty:
        raise ValueError("测试集无可用预测结果。")

    target_symbol = args.symbol.upper()
    if target_symbol not in pred_df["symbol"].unique():
        raise ValueError(f"预测结果中不存在标的 {target_symbol}，可用标的：{sorted(pred_df['symbol'].unique())}")

    price_df = frames["test"]
    price_symbol_df = price_df[price_df["symbol"] == target_symbol][["date", "open", "high", "low", "close"]].copy()
    if price_symbol_df.empty:
        raise ValueError(f"价格数据中不存在标的 {target_symbol}")

    symbol_pred_df = pred_df[pred_df["symbol"] == target_symbol]
    output_path = Path(args.output) if args.output else run_dir / f"inference_{model_name}_{feature_mode}_{target_symbol}.png"
    plot_candlestick_with_predictions(price_symbol_df, symbol_pred_df, target_symbol, output_path)
    print(f"[Done] 图像已保存：{output_path}")


if __name__ == "__main__":
    main()
