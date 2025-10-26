from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

EXPERIMENT_ROOT = Path(__file__).resolve().parent

from data_pipeline import (
    apply_scaler,
    build_dataloaders,
    fit_scaler,
    prepare_feature_dataframe,
    save_scaler,
    split_by_date,
)
from models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train trend-scan return regression model")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    mse_fn = nn.MSELoss(reduction="sum")
    mae_fn = nn.L1Loss(reduction="sum")
    mse_total = 0.0
    mae_total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            mse_total += mse_fn(preds, targets).item()
            mae_total += mae_fn(preds, targets).item()
            count += targets.numel()
    if count == 0:
        return float("nan"), float("nan")
    return mse_total / count, mae_total / count


def train_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer,
    criterion,
    grad_clip: float | None = None,
) -> Tuple[float, float]:
    model.train()
    mse_fn = nn.MSELoss(reduction="sum")
    mae_fn = nn.L1Loss(reduction="sum")
    mse_total = 0.0
    mae_total = 0.0
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_mse = mse_fn(preds.detach(), targets).item()
        batch_mae = mae_fn(preds.detach(), targets).item()
        mse_total += batch_mse
        mae_total += batch_mae
        count += targets.numel()
    if count == 0:
        return float("nan"), float("nan")
    return mse_total / count, mae_total / count


def plot_history(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_mse"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(epochs, history["train_mse"], label="Train MSE", color="#1f77b4")
    axes[0].plot(epochs, history["val_mse"], label="Val MSE", color="#ff7f0e")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(epochs, history["train_mae"], label="Train MAE", color="#2ca02c")
    axes[1].plot(epochs, history["val_mae"], label="Val MAE", color="#d62728")
    axes[1].set_ylabel("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    feature_mode = config.get("feature_mode", "alpha")
    feature_mode_params = dict(config.get("feature_mode_params", {}))
    feature_mode_params.setdefault("alpha_kind", "alpha158")
    feature_mode_params.setdefault("min_r2", 0.75)
    window_size = int(config.get("window_size", 96))
    batch_size = int(config.get("batch_size", 128))
    num_workers = int(config.get("num_workers", 0))
    splits = config.get(
        "splits",
        {
            "train": ["2014-01-01", "2021-12-31"],
            "val": ["2022-01-01", "2022-12-31"],
            "test": ["2023-01-01", "2023-12-31"],
        },
    )

    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 5e-5))
    epochs = int(config.get("epochs", 120))
    grad_clip = config.get("grad_clip", 1.0)
    checkpoint_every = 10

    output_dir = Path(config.get("output_dir", EXPERIMENT_ROOT / "outputs" / "training"))
    ensure_output_dir(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    print(f"[Data] feature_mode={feature_mode}, window={window_size}, batch={batch_size}")
    if feature_mode_params:
        print(f"[Data] feature_mode_params={feature_mode_params}")
    frame = prepare_feature_dataframe(feature_mode, mode_params=feature_mode_params)
    frames = split_by_date(frame, splits)

    train_frame = frames.get("train", pd.DataFrame())
    if train_frame.empty:
        raise ValueError("训练集为空，请检查时间区间或特征准备流程。")

    exclude_cols = {"label", "symbol", "date"}
    feature_cols = [col for col in train_frame.columns if col not in exclude_cols and not col.startswith("trend_")]

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "transformer")
    model_params = model_cfg.get("params", {})

    scaler = fit_scaler(train_frame, feature_cols)
    scaler_path = output_dir / f"scaler_{model_name}_{feature_mode}.json"
    save_scaler(scaler_path, scaler)

    scaled_frames = {name: apply_scaler(df, scaler) for name, df in frames.items()}
    feature_cols = [col for col in scaled_frames["train"].columns if col not in exclude_cols and not col.startswith("trend_")]

    loaders = build_dataloaders(scaled_frames, feature_cols, window_size, batch_size, num_workers=num_workers)

    model = create_model(model_name, input_dim=len(feature_cols), seq_len=window_size, **model_params)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 10))
    criterion = nn.L1Loss()

    history = {
        "train_mse": [],
        "train_mae": [],
        "val_mse": [],
        "val_mae": [],
    }

    best_val_mae = float("inf")
    best_val_mse = float("inf")
    best_epoch = -1
    best_path = output_dir / "checkpoints" / f"best_{model_name}_{feature_mode}.pt"
    initial_train_mae = None
    guard_threshold = None
    guard_released = False

    for epoch in range(1, epochs + 1):
        train_mse, train_mae = train_epoch(model, loaders["train"], device, optimizer, criterion, grad_clip)
        val_mse, val_mae = evaluate(model, loaders["val"], device)
        history["train_mse"].append(train_mse)
        history["train_mae"].append(train_mae)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        scheduler.step()

        if initial_train_mae is None and np.isfinite(train_mae):
            initial_train_mae = train_mae
            guard_threshold = initial_train_mae * 0.25
        ready_for_best = guard_threshold is not None and train_mae <= guard_threshold
        if ready_for_best and not guard_released:
            guard_released = True
            print(
                f"[Guard] Unlocked checkpointing (train_mae={train_mae:.4f} <= {guard_threshold:.4f})"
            )

        print(
            f"[Epoch {epoch:03d}/{epochs}] train_mse={train_mse:.4f} train_mae={train_mae:.4f} "
            f"val_mse={val_mse:.4f} val_mae={val_mae:.4f}"
        )

        if ready_for_best and val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, best_path)

        if epoch % checkpoint_every == 0 or epoch == epochs:
            ckpt_path = output_dir / "checkpoints" / f"{model_name}_{feature_mode}_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                },
                ckpt_path,
            )

    metrics_path = output_dir / f"metrics_{model_name}_{feature_mode}.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    plot_history(history, output_dir / f"training_curves_{model_name}_{feature_mode}.png")

    if best_epoch > 0:
        print(f"[Result] Best val MAE {best_val_mae:.4f} at epoch {best_epoch}")
        checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
    else:
        best_path = None

    test_mse, test_mae = evaluate(model, loaders["test"], device)
    print(f"[Test] mse={test_mse:.4f} mae={test_mae:.4f}")

    metadata = {
        "config_path": str(config_path),
        "feature_mode": feature_mode,
        "feature_mode_params": feature_mode_params,
        "feature_cols": feature_cols,
        "window_size": window_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_val_mse": best_val_mse,
        "initial_train_mae": initial_train_mae,
        "best_guard_threshold": guard_threshold,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "model_name": model_name,
        "model_params": model_params,
        "scaler_path": str(scaler_path.resolve()),
        "best_checkpoint": str(best_path.resolve()) if best_path else "",
        "splits": splits,
    }
    (output_dir / f"metadata_{model_name}_{feature_mode}.json").write_text(json.dumps(metadata, indent=2))

    summary_lines = [
        f"best_epoch: {best_epoch if best_epoch > 0 else 'N/A'}",
        f"best_val_mae: {best_val_mae:.6f}" if np.isfinite(best_val_mae) else "best_val_mae: N/A",
        f"best_val_mse: {best_val_mse:.6f}" if np.isfinite(best_val_mse) else "best_val_mse: N/A",
        f"test_mae: {test_mae:.6f}" if np.isfinite(test_mae) else "test_mae: N/A",
        f"test_mse: {test_mse:.6f}" if np.isfinite(test_mse) else "test_mse: N/A",
        f"initial_train_mae: {initial_train_mae:.6f}" if initial_train_mae else "initial_train_mae: N/A",
        f"guard_threshold: {guard_threshold:.6f}" if guard_threshold else "guard_threshold: N/A",
    ]
    (output_dir / "training_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[Done] Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
