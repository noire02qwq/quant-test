from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch

from data_pipeline import (
    apply_scaler,
    build_dataloaders,
    load_scaler,
    prepare_feature_dataframe,
    split_by_date,
)
from models import create_model
from train import EXPERIMENT_ROOT

TARGET_SPLITS: Sequence[str] = ("val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use best checkpoint to infer and plot regression results on val+test.")
    parser.add_argument("--config", required=True, help="训练时使用的配置 JSON 路径。")
    parser.add_argument("--metadata", default=None, help="训练输出的 metadata JSON，默认自动推断。")
    parser.add_argument("--checkpoint", default=None, help="使用的模型权重，默认 metadata 中 best_checkpoint。")
    parser.add_argument("--symbol", default="TSM", help="绘图使用的标的（默认 TSM）。")
    parser.add_argument("--output", default=None, help="输出图像路径，默认写入 run 目录。")
    parser.add_argument("--device", default=None, help="推理设备（cpu/cuda），默认与训练一致。")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def gather_split_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split_name: str,
) -> pd.DataFrame:
    dataset = loader.dataset
    if len(dataset) == 0:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    processed = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().squeeze(-1)
            batch_y = batch_y.cpu().squeeze(-1)
            batch_size = preds.shape[0]
            for i in range(batch_size):
                idx = processed + i
                records.append(
                    {
                        "symbol": dataset.symbols[idx],
                        "date": pd.Timestamp(dataset.timestamps[idx]),
                        "label": float(batch_y[i].item()),
                        "prediction": float(preds[i].item()),
                        "split": split_name,
                    }
                )
            processed += batch_size
    return pd.DataFrame.from_records(records)


def build_price_frame(frames: Dict[str, pd.DataFrame], splits: Iterable[str], symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    clips: List[pd.DataFrame] = []
    for split in splits:
        if split not in frames:
            continue
        frame = frames[split]
        if frame.empty:
            continue
        clip = frame[frame["symbol"] == symbol].copy()
        if clip.empty:
            continue
        clip = clip[(clip["date"] >= start_date) & (clip["date"] <= end_date)]
        if clip.empty:
            continue
        clip = clip[["date", "open", "high", "low", "close", "label"]].copy()
        clip["split"] = split
        clips.append(clip)
    if not clips:
        raise ValueError(f"{symbol} 在指定切分 {list(splits)} 的区间内没有可用数据。")
    merged = pd.concat(clips, axis=0).sort_values("date").reset_index(drop=True)
    merged = merged.rename(columns={"label": "actual_label"})
    return merged


def plot_regression_results(price_df: pd.DataFrame, pred_df: pd.DataFrame, symbol: str, output_path: Path) -> None:
    if pred_df.empty:
        raise ValueError("预测结果为空，无法绘图。")

    pred_symbol = pred_df[pred_df["symbol"] == symbol].copy()
    if pred_symbol.empty:
        raise ValueError(f"预测结果中不存在标的 {symbol}。")

    merged = price_df.merge(pred_symbol[["date", "prediction", "split"]], on="date", how="inner")
    if merged.empty:
        raise ValueError("价格数据与预测无法对齐，请检查数据处理流程。")
    merged = merged.sort_values("date").reset_index(drop=True)

    positions = np.arange(len(merged))

    fig, (ax_price, ax_actual, ax_pred) = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2, 1.2]},
    )

    candle_width = 0.6
    for idx, row in merged.iterrows():
        color = "red" if row["close"] >= row["open"] else "green"
        ax_price.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1)
        lower = min(row["open"], row["close"])
        height = max(row["open"], row["close"]) - lower or 1e-10
        ax_price.add_patch(Rectangle((idx - candle_width / 2, lower), candle_width, height, edgecolor=color, facecolor=color))

    ax_price.set_title(f"{symbol} Val+Test K-line")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)

    ax_actual.plot(positions, merged["actual_label"], label="Actual Label", color="#1f77b4")
    ax_actual.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
    ax_actual.set_ylabel("Actual")
    ax_actual.grid(True, linestyle="--", alpha=0.3)
    ax_actual.legend(loc="upper left")

    ax_pred.plot(positions, merged["prediction"], label="Prediction", color="#d62728", linestyle="--")
    ax_pred.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
    ax_pred.set_ylabel("Predicted")
    ax_pred.set_xlabel("Trade Date")
    ax_pred.grid(True, linestyle="--", alpha=0.3)
    ax_pred.legend(loc="upper left")

    tick_step = max(len(merged) // 12, 1)
    tick_positions = list(range(0, len(merged), tick_step))
    if tick_positions[-1] != len(merged) - 1:
        tick_positions.append(len(merged) - 1)
    tick_labels = [merged.iloc[pos]["date"].strftime("%Y-%m-%d") for pos in tick_positions]
    ax_pred.set_xticks(tick_positions)
    ax_pred.set_xticklabels(tick_labels, rotation=45, ha="right")

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

    frame = prepare_feature_dataframe(feature_mode, symbols=symbols if symbols else None)
    frames = split_by_date(frame, splits)

    val_frame = frames.get("val")
    test_frame = frames.get("test")
    if val_frame is None or val_frame.empty:
        raise ValueError("验证集为空，无法确定可视化区间，请检查配置。")
    if test_frame is None or test_frame.empty:
        raise ValueError("测试集为空，无法确定可视化区间，请检查配置。")
    start_date = pd.to_datetime(val_frame["date"]).min()
    end_date = pd.to_datetime(test_frame["date"]).max()
    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError("无法获取验证或测试集的日期范围。")

    scaler = load_scaler(Path(metadata["scaler_path"]))
    scaled_frames = {name: apply_scaler(df, scaler) for name, df in frames.items()}
    feature_cols = metadata.get("feature_cols")
    if not feature_cols:
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

    target_symbol = args.symbol.upper()
    available_splits = [split for split in TARGET_SPLITS if split in loaders]
    if not available_splits:
        raise ValueError("未找到 val/test DataLoader，请确认配置中的切分。")

    pred_frames = []
    for split in available_splits:
        loader = loaders[split]
        if len(loader.dataset) == 0:
            continue
        split_pred = gather_split_predictions(model, loader, device, split)
        if not split_pred.empty:
            pred_frames.append(split_pred)
    if not pred_frames:
        raise ValueError("val/test 均无可用预测结果。")
    pred_df = pd.concat(pred_frames, axis=0)
    pred_df = pred_df[(pred_df["date"] >= start_date) & (pred_df["date"] <= end_date)]
    pred_df = pred_df.sort_values(["date", "symbol"]).reset_index(drop=True)
    if pred_df.empty:
        raise ValueError("在指定的可视化区间内没有模型预测结果。")

    price_df = build_price_frame(frames, available_splits, target_symbol, start_date, end_date)

    combined = price_df.merge(pred_df[pred_df["symbol"] == target_symbol][["date", "prediction"]], on="date", how="inner")
    mse = float(np.mean((combined["actual_label"].to_numpy() - combined["prediction"].to_numpy()) ** 2))
    mae = float(np.mean(np.abs(combined["actual_label"].to_numpy() - combined["prediction"].to_numpy())))

    output_path = Path(args.output) if args.output else run_dir / f"inference_{model_name}_{feature_mode}_{target_symbol}_val_test.png"
    plot_regression_results(price_df, pred_df, target_symbol, output_path)
    print(f"[Metrics] Val+Test MSE={mse:.4f} MAE={mae:.4f}")
    print(f"[Done] 图像已保存：{output_path}")


if __name__ == "__main__":
    main()
