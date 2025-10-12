from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser(description="Train regression model for experiment-1010")
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
            count += targets.size(0)
    if count == 0:
        return float("nan"), float("nan")
    return mse_total / count, mae_total / count


def train_epoch(model: nn.Module, loader, device: torch.device, optimizer, criterion, grad_clip: float | None = None) -> Tuple[float, float]:
    model.train()
    mae_fn = nn.L1Loss(reduction="sum")
    total_loss = 0.0
    total_mae = 0.0
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
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_mae += mae_fn(preds.detach(), targets).item()
        count += batch_size
    return total_loss / count, total_mae / count


def plot_history(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(epochs, history["train_loss"], label="Train MSE", color="#1f77b4")
    axes[0].set_ylabel("Train Loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].plot(epochs, history["val_mse"], label="Val MSE", color="#ff7f0e")
    axes[1].set_ylabel("Val MSE")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].plot(epochs, history["val_mae"], label="Val MAE", color="#2ca02c")
    axes[2].set_ylabel("Val MAE")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    for ax in axes:
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    feature_mode = config.get("feature_mode", "raw_traditional")
    window_size = int(config.get("window_size", 60))
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 0))
    splits = config.get(
        "splits",
        {
            "train": ["2015-01-01", "2022-12-31"],
            "val": ["2023-01-01", "2023-12-31"],
            "test": ["2024-01-01", "2024-12-31"],
        },
    )

    learning_rate = float(config.get("learning_rate", 5e-4))
    weight_decay = float(config.get("weight_decay", 1e-5))
    epochs = int(config.get("epochs", 60))
    grad_clip = config.get("grad_clip", 1.0)
    checkpoint_every = int(config.get("checkpoint_every", 10))

    output_dir = Path(config.get("output_dir", EXPERIMENT_ROOT / "outputs" / "training"))
    ensure_output_dir(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    print(f"[Data] feature_mode={feature_mode}, window={window_size}, batch={batch_size}")
    frame = prepare_feature_dataframe(feature_mode)
    frames = split_by_date(frame, splits)

    train_frame = frames["train"]
    if train_frame.empty:
        raise ValueError("训练集为空，请检查时间区间或特征准备流程。")

    feature_cols = [col for col in train_frame.columns if col not in {"label", "symbol", "date"}]
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "transformer")
    model_params = model_cfg.get("params", {})

    scaler = fit_scaler(train_frame, feature_cols)
    scaler_path = output_dir / f"scaler_{model_name}_{feature_mode}.json"
    save_scaler(scaler_path, scaler)

    scaled_frames = {name: apply_scaler(df, scaler) for name, df in frames.items()}
    feature_cols = [col for col in scaled_frames["train"].columns if col not in {"label", "symbol", "date"}]

    loaders = build_dataloaders(scaled_frames, feature_cols, window_size, batch_size, num_workers=num_workers)

    model = create_model(model_name, input_dim=len(feature_cols), seq_len=window_size, **model_params)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 10))
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "train_mae": [],
        "val_mse": [],
        "val_mae": [],
    }

    best_val_mse = float("inf")
    best_epoch = -1
    best_path = output_dir / "checkpoints" / f"best_{model_name}_{feature_mode}.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_mae = train_epoch(model, loaders["train"], device, optimizer, criterion, grad_clip)
        val_mse, val_mae = evaluate(model, loaders["val"], device)
        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_mse"].append(val_mse)
        history["val_mae"].append(val_mae)

        scheduler.step()

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_mae={train_mae:.4f} "
            f"val_mse={val_mse:.4f} val_mae={val_mae:.4f}"
        )

        if val_mse < best_val_mse:
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
        print(f"[Result] Best validation MSE {best_val_mse:.4f} at epoch {best_epoch}")
        checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
    else:
        best_path = None

    test_mse, test_mae = evaluate(model, loaders["test"], device)
    print(f"[Test] test_mse={test_mse:.4f} test_mae={test_mae:.4f}")

    metadata = {
        "config_path": str(config_path),
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "window_size": window_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "model_name": model_name,
        "model_params": model_params,
        "scaler_path": str(scaler_path.resolve()),
        "best_checkpoint": str(best_path.resolve()) if best_path else "",
        "splits": splits,
    }
    (output_dir / f"metadata_{model_name}_{feature_mode}.json").write_text(json.dumps(metadata, indent=2))
    print(f"[Done] Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()

# python experiment-1010/train.py --config experiment-1010/configs/transformer_raw_traditional.json
