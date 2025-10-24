from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from data_module import create_dataloaders, prepare_datasets, ScalerState
from models import MODEL_REGISTRY, TransformerConfig


def load_config(path: Path) -> Dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["paths"] = cfg.get("paths", {})
    return cfg


def build_scaler_state(scaler_dict: Dict) -> ScalerState:
    columns = scaler_dict["columns"]
    mean = pd.Series(scaler_dict["mean"], index=columns, dtype=np.float32)
    std = pd.Series(scaler_dict["std"], index=columns, dtype=np.float32)
    return ScalerState(mean=mean, std=std)


def win_rate_score(y_true: torch.Tensor, y_pred: torch.Tensor, positive_class: int = 1) -> float:
    mask = y_pred == positive_class
    positives = mask.sum().item()
    if positives == 0:
        return float("nan")
    correct = ((y_true == positive_class) & mask).sum().item()
    return correct / positives


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            pred = logits.argmax(dim=1)
            targets.append(labels.cpu())
            preds.append(pred.cpu())
    if not targets:
        return {"accuracy": float("nan"), "win_rate": float("nan"), "macro_f1": float("nan"), "samples": 0}
    y_true = torch.cat(targets)
    y_pred = torch.cat(preds)
    acc = (y_true == y_pred).float().mean().item()
    win_rate = win_rate_score(y_true, y_pred)
    scores = []
    for cls in range(num_classes):
        tp = ((y_pred == cls) & (y_true == cls)).sum().item()
        fp = ((y_pred == cls) & (y_true != cls)).sum().item()
        fn = ((y_pred != cls) & (y_true == cls)).sum().item()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        scores.append(2 * precision * recall / (precision + recall + 1e-12))
    f1 = float(np.mean(scores))
    return {"accuracy": acc, "win_rate": win_rate, "macro_f1": f1, "samples": len(y_true)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transformer checkpoints for preposition-experiments/experiment-1005")
    parser.add_argument("--config", type=str, default="preposition-experiments/experiment-1005/configs/transformer_alpha158.json")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    config = load_config(cfg_path)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is None:
        paths_cfg = config.get("paths", {})
        checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "preposition-experiments/experiment-1005/outputs/checkpoints"))
        feature_tag = config["data"]["feature_set"]
        fallback = checkpoint_dir / f"{config.get('experiment_name', f'exp1005_{feature_tag}')}_best.pt"
        checkpoint_path = fallback
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = payload.get("config", config)
    feature_cols_saved = payload["feature_cols"]
    scaler_state = build_scaler_state(payload["scaler"])

    ds_train, ds_val, ds_test, feature_cols, _, _ = prepare_datasets(
        config["data"], scaler_override=scaler_state, feature_cols_override=feature_cols_saved
    )
    train_symbol_count = len({sym for sym, _ in ds_train.meta})
    val_symbol_count = len({sym for sym, _ in ds_val.meta})
    test_symbol_count = len({sym for sym, _ in ds_test.meta})
    print(
        f"[preposition-experiments/experiment-1005] Dataset (eval) symbol coverage -> train: {train_symbol_count}, val: {val_symbol_count}, test: {test_symbol_count}"
    )
    batch_size = args.batch_size
    train_loader, val_loader, test_loader = create_dataloaders(ds_train, ds_val, ds_test, batch_size)
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataset_map = {"train": ds_train, "val": ds_val, "test": ds_test}

    model_name = str(ckpt_config.get("model", {}).get("name", "transformer")).lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' unavailable in registry.")
    meta = payload.get("meta", {})
    if "num_classes" in meta:
        num_classes = int(meta["num_classes"])
    else:
        head_key = next((k for k in payload["state_dict"].keys() if k.endswith(".weight") and "head" in k), None)
        if head_key is None:
            raise KeyError("Unable to infer number of classes from checkpoint; missing head weight.")
        num_classes = payload["state_dict"][head_key].shape[0]
    model_cfg = ckpt_config.get("model", {})
    transformer_cfg = TransformerConfig(
        feature_dim=len(feature_cols_saved),
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
    model = MODEL_REGISTRY[model_name](transformer_cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)

    loader = loader_map[args.split]
    metrics = evaluate(model, loader, device, num_classes)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print(f"Samples: {metrics['samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} (overall correctness)")
    print(f"Win Rate: {metrics['win_rate']:.4f} (precision for positive/win predictions)")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
