from __future__ import annotations

import argparse
import json
import random
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Sequence, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_module import prepare_datasets, ScalerState, resolve_symbols
from models import MODEL_REGISTRY, TransformerConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).float().mean().item()


def win_rate_score(y_true: torch.Tensor, y_pred: torch.Tensor, positive_class: int = 1) -> float:
    mask = y_pred == positive_class
    positives = mask.sum().item()
    if positives == 0:
        return float("nan")
    correct = ((y_true == positive_class) & mask).sum().item()
    return correct / positives


def macro_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    scores = []
    for cls in range(num_classes):
        tp = ((y_pred == cls) & (y_true == cls)).sum().item()
        fp = ((y_pred == cls) & (y_true != cls)).sum().item()
        fn = ((y_pred != cls) & (y_true == cls)).sum().item()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        scores.append(2 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(scores))


def load_config(path: Path) -> Dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["paths"] = cfg.get("paths", {})
    return cfg


def compute_class_weights(labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights.to(device)


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler_state: ScalerState,
    feature_cols: Sequence[str],
    config: Dict,
    meta: Dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler": scaler_state.as_dict(),
        "feature_cols": list(feature_cols),
        "config": config,
        "meta": meta,
    }
    torch.save(payload, checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transformer for preposition-experiments/experiment-1005")
    parser.add_argument("--config", type=str, default="preposition-experiments/experiment-1005/configs/transformer_alpha158.json")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    config = load_config(cfg_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = pick_device(str(config.get("device", "auto")))

    paths_cfg = config.get("paths", {})
    output_root = Path(paths_cfg.get("output_dir", "preposition-experiments/experiment-1005/outputs")).resolve()
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", output_root / "checkpoints")).resolve()
    log_dir = Path(paths_cfg.get("log_dir", output_root / "logs")).resolve()
    for directory in [output_root, checkpoint_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    data_cfg_base = deepcopy(config["data"])
    min_start = pd.Timestamp("2010-01-01")
    min_end = pd.Timestamp("2024-12-31")
    all_symbols = resolve_symbols(
        data_cfg_base.get("symbols"),
        data_cfg_base.get("raw_data_dir"),
        min_start=min_start,
        min_end=min_end,
    )
    if not all_symbols:
        raise RuntimeError("No symbols available for training. Check raw data directory or configuration.")
    print(f"[preposition-experiments/experiment-1005] Total symbols available: {len(all_symbols)}")

    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    num_workers = int(training_cfg.get("num_workers", 0))
    def load_group_dataset(symbols: Sequence[str]):
        group_cfg = deepcopy(data_cfg_base)
        group_cfg["symbols"] = list(symbols)
        group_cfg["shuffle_symbols"] = False
        return prepare_datasets(group_cfg, quiet=True)

    model_name = str(config.get("model", {}).get("name", "transformer")).lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model_cfg = config.get("model", {})
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    criterion: nn.Module | None = None
    feature_cols_ref: List[str] | None = None
    current_feature_cols: List[str] | None = None
    current_scaler: ScalerState | None = None
    num_classes: int | None = None

    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    epochs = int(training_cfg["epochs"])
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 5))
    experiment_name = config.get("experiment_name", f"exp1005_{data_cfg_base['feature_set']}")

    best_metrics: Dict[str, float] = {"val_f1": -1.0, "val_acc": -1.0, "val_win_rate": -1.0}
    best_state = None
    best_checkpoint_path = checkpoint_dir / f"{experiment_name}_best.pt"
    global_step = 0

    def _to_int(val) -> int | None:
        if val is None:
            return None
        try:
            if isinstance(val, str):
                val = val.strip()
                if not val:
                    return None
            intval = int(val)
        except (TypeError, ValueError):
            return None
        return intval

    group_size_cfg_value = _to_int(training_cfg.get("symbol_group_size"))
    if group_size_cfg_value is not None and group_size_cfg_value <= 0:
        group_size_cfg_value = None
    group_count_cfg_value = _to_int(training_cfg.get("symbol_group_count"))
    if group_count_cfg_value is not None and group_count_cfg_value <= 0:
        group_count_cfg_value = None

    symbol_checkpoint_interval = _to_int(training_cfg.get("checkpoint_symbol_interval", 50)) or 50
    symbol_batch_size = _to_int(training_cfg.get("symbol_batch_size", 5)) or 5

    avg_train_loss = float("nan")
    test_acc = float("nan")
    test_win_rate = float("nan")
    test_f1 = float("nan")

    def aggregate_metrics(metrics: List[tuple[float, float, float, int]]) -> tuple[float, float, float]:
        if not metrics:
            return (float("nan"), float("nan"), float("nan"))
        sums = np.zeros(3, dtype=float)
        counts = np.zeros(3, dtype=float)
        for acc, win_rate, f1, weight in metrics:
            for idx, val in enumerate((acc, win_rate, f1)):
                if not math.isnan(val):
                    sums[idx] += val * weight
                    counts[idx] += weight
        return tuple(sums[i] / counts[i] if counts[i] > 0 else float("nan") for i in range(3))

    def weighted_mean(values: List[tuple[float, int]]) -> float:
        if not values:
            return float("nan")
        total_sum = 0.0
        total_weight = 0.0
        for val, weight in values:
            if not math.isnan(val):
                total_sum += val * weight
                total_weight += weight
        return total_sum / total_weight if total_weight > 0 else float("nan")

    def train_and_evaluate_group(
        epoch: int,
        current_symbols_list: Sequence[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> tuple[float, float, float, float]:
        nonlocal global_step, best_state, avg_train_loss, test_acc, test_win_rate, test_f1

        total_loss = 0.0
        total_samples = 0
        for features, labels in train_loader:
            global_step += 1
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            batch_size_actual = features.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

        avg_train_loss = total_loss / total_samples if total_samples > 0 else float("nan")

        model.eval()
        val_targets, val_preds = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                pred = logits.argmax(dim=1)
                val_targets.append(labels.cpu())
                val_preds.append(pred.cpu())
        if val_targets:
            y_true = torch.cat(val_targets)
            y_pred = torch.cat(val_preds)
            val_acc = accuracy_score(y_true, y_pred)
            val_win_rate = win_rate_score(y_true, y_pred)
            val_f1 = macro_f1_score(y_true, y_pred, num_classes or 2)
        else:
            val_acc = val_win_rate = val_f1 = float("nan")

        test_targets, test_preds = [], []
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                pred = logits.argmax(dim=1)
                test_targets.append(labels.cpu())
                test_preds.append(pred.cpu())
        if test_targets:
            y_true_test = torch.cat(test_targets)
            y_pred_test = torch.cat(test_preds)
            test_acc_local = accuracy_score(y_true_test, y_pred_test)
            test_win_rate_local = win_rate_score(y_true_test, y_pred_test)
            test_f1_local = macro_f1_score(y_true_test, y_pred_test, num_classes or 2)
        else:
            test_acc_local = test_win_rate_local = test_f1_local = float("nan")

        test_acc = test_acc_local
        test_win_rate = test_win_rate_local
        test_f1 = test_f1_local

        return val_acc, val_win_rate, val_f1, avg_train_loss

    for epoch in range(1, epochs + 1):
        epoch_symbols = all_symbols.copy()
        random.shuffle(epoch_symbols)
        val_metrics_epoch: List[tuple[float, float, float]] = []
        train_losses_epoch: List[float] = []
        test_metrics_epoch: List[tuple[float, float, float]] = []

        print(f"[preposition-experiments/experiment-1005] Epoch {epoch}: processing {len(epoch_symbols)} symbols")

        batch_val_metrics: List[tuple[float, float, float]] = []

        for idx, symbol in enumerate(epoch_symbols, start=1):
            ds_train, ds_val, ds_test, feature_cols, scaler, _ = load_group_dataset([symbol])
            if len(ds_train) == 0 or len(ds_val) == 0 or len(ds_test) == 0:
                print(f"[preposition-experiments/experiment-1005] Skipping {symbol} due to insufficient data")
                continue

            if feature_cols_ref is None:
                feature_cols_ref = feature_cols
                current_feature_cols = feature_cols
                current_scaler = scaler
                num_classes = max(2, int(ds_train.y.max().item()) + 1)
                transformer_cfg = TransformerConfig(
                    feature_dim=len(feature_cols_ref),
                    num_classes=num_classes,
                    d_model=int(model_cfg.get("d_model", 128)),
                    nhead=int(model_cfg.get("nhead", 4)),
                    num_layers=int(model_cfg.get("num_layers", 4)),
                    dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
                    dropout=float(model_cfg.get("dropout", 0.1)),
                    pooling=str(model_cfg.get("pooling", "mean")),
                    norm_first=bool(model_cfg.get("norm_first", True)),
                    embedding_dropout=float(model_cfg.get("embedding_dropout", 0.05)),
                )
                model = MODEL_REGISTRY[model_name](transformer_cfg).to(device)
                optimizer = AdamW(
                    model.parameters(),
                    lr=float(training_cfg["learning_rate"]),
                    weight_decay=float(training_cfg.get("weight_decay", 0.0)),
                    betas=tuple(training_cfg.get("betas", (0.9, 0.999))),
                )
                print("Model configuration:")
                for key, value in model.describe().items():
                    print(f"  {key}: {value}")
                print("Note: validation/test `acc` represents overall accuracy (correct predictions / total samples).")
            else:
                if len(feature_cols) != len(feature_cols_ref):
                    raise ValueError("Feature dimensionality changed between symbols; ensure preprocessing consistency.")
                current_feature_cols = feature_cols
                current_scaler = scaler

            class_weights = compute_class_weights(ds_train.y, num_classes or 2, device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
            val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
            test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

            val_acc, val_win_rate, val_f1, train_loss_symbol = train_and_evaluate_group(
                epoch, [symbol], train_loader, val_loader, test_loader
            )
            val_metrics_epoch.append((val_acc, val_win_rate, val_f1))
            batch_val_metrics.append((val_acc, val_win_rate, val_f1))
            train_losses_epoch.append(train_loss_symbol)
            test_metrics_epoch.append((test_acc, test_win_rate, test_f1))

            if idx % symbol_checkpoint_interval == 0:
                val_avg = aggregate_metrics(batch_val_metrics)
                if current_scaler is not None and current_feature_cols is not None:
                    save_checkpoint(
                        checkpoint_dir / f"{experiment_name}_feature-{data_cfg_base['feature_set']}_epoch-{epoch}_symbol-{idx}.pt",
                        model,
                        optimizer,
                        current_scaler,
                        current_feature_cols,
                        config,
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "feature_set": data_cfg_base["feature_set"],
                            "checkpoint_type": "symbol",
                            "processed_symbols": epoch_symbols[:idx],
                            "val_acc": val_avg[0],
                            "val_win_rate": val_avg[1],
                            "val_f1": val_avg[2],
                            "num_classes": num_classes or 2,
                        },
                    )
                    print(
                        f"[preposition-experiments/experiment-1005] Epoch {epoch} checkpoint after {idx} symbols: "
                        f"val_acc={val_avg[0]:.4f}, val_win_rate={val_avg[1]:.4f}, val_f1={val_avg[2]:.4f}"
                    )
                batch_val_metrics.clear()

        val_epoch_avg = aggregate_metrics(val_metrics_epoch)
        test_epoch_avg = aggregate_metrics(test_metrics_epoch)

        print(
            f"Epoch {epoch:03d} summary | train_loss={np.nanmean(train_losses_epoch):.6f} | "
            f"val_acc={val_epoch_avg[0]:.4f} | val_win_rate={val_epoch_avg[1]:.4f} | val_f1={val_epoch_avg[2]:.4f}"
        )

        if val_epoch_avg[2] > best_metrics["val_f1"]:
            best_metrics["val_f1"] = val_epoch_avg[2]
            best_metrics["val_acc"] = val_epoch_avg[0]
            best_metrics["val_win_rate"] = val_epoch_avg[1]
            best_metrics["epoch"] = epoch
            best_metrics["global_step"] = global_step
            best_state = deepcopy(model.state_dict()) if model is not None else None
            if current_scaler is not None and current_feature_cols is not None:
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    current_scaler,
                    current_feature_cols,
                    config,
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "feature_set": data_cfg_base["feature_set"],
                        "checkpoint_type": "best",
                        "val_acc": val_epoch_avg[0],
                        "val_win_rate": val_epoch_avg[1],
                        "val_f1": val_epoch_avg[2],
                        "num_classes": num_classes or 2,
                        "symbols": epoch_symbols,
                    },
                )
                print(f"  Saved new best checkpoint: {best_checkpoint_path}")

        test_acc, test_win_rate, test_f1 = test_epoch_avg

    print(
        f"Best Epoch {best_metrics.get('epoch')} | val_acc={best_metrics.get('val_acc'):.4f} | "
        f"val_win_rate={best_metrics.get('val_win_rate'):.4f} | val_f1={best_metrics.get('val_f1'):.4f}"
    )
    print(
        f"Test metrics | acc={test_acc:.4f} | win_rate={test_win_rate:.4f} | macro_f1={test_f1:.4f}"
    )

    summary = {
        "experiment_name": experiment_name,
        "feature_set": data_cfg_base["feature_set"],
        "best_epoch": best_metrics.get("epoch"),
        "best_val_acc": best_metrics.get("val_acc"),
        "best_val_win_rate": best_metrics.get("val_win_rate"),
        "best_val_f1": best_metrics.get("val_f1"),
        "test_acc": test_acc,
        "test_win_rate": test_win_rate,
        "test_f1": test_f1,
        "total_steps": global_step,
        "train_loss_last": float(avg_train_loss),
    }
    summary_path = log_dir / f"{experiment_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
