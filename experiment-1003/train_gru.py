"""GRU 训练脚本：沿用现有数据加载与特征构建，替换模型为 GRU。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import load_dataframe, prepare_frames, compute_scaler, apply_scaler, SeriesConfig, WindowDataset, make_loaders
from gru_model import GRUClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).float().mean().item()


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 3) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))

def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    cfg: Dict = json.loads(Path("experiment-1003/config.json").read_text(encoding="utf-8"))
    set_seed(int(cfg["random_seed"]))
    device = pick_device(str(cfg.get("device", "auto")))

    data_path = Path(cfg["data_path"])
    out_dir = Path(cfg["output_dir"]) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(data_path, index_col=cfg.get("index_col", "datetime"))
    train_df, val_df, test_df, feature_cols = prepare_frames(df, cfg)
    # 仅基于训练集计算标准化参数，并应用到三段数据
    scaler = compute_scaler(train_df, feature_cols)
    train_df = apply_scaler(train_df, scaler)
    val_df = apply_scaler(val_df, scaler)
    test_df = apply_scaler(test_df, scaler)
    # 构建数据集/加载器
    series_cfg = SeriesConfig(window_size=int(cfg["window_size"]), feature_cols=feature_cols)
    ds_train = WindowDataset(train_df, series_cfg)
    ds_val = WindowDataset(val_df, series_cfg)
    ds_test = WindowDataset(test_df, series_cfg)
    train_loader, val_loader, _ = make_loaders(ds_train, ds_val, ds_test, batch_size=int(cfg["batch_size"]))

    model = GRUClassifier(
        feature_dim=len(feature_cols),
        hidden_size=256,
        num_layers=3,
        dropout=0.1,
        bidirectional=False,
        pooling="last",
        num_classes=3,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    # 类别不平衡：基于训练集 y 分布设置权重
    class_counts = torch.bincount(ds_train.y)
    class_counts = class_counts.float().clamp(min=1.0)
    weights = (class_counts.sum() / (len(class_counts) * class_counts))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_f1 = -1.0
    best_acc = -1.0
    best_path = None
    for epoch in range(1, int(cfg["num_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                pred = logits.argmax(dim=1)
                all_y.append(y.cpu())
                all_p.append(pred.cpu())
        y_true = torch.cat(all_y)
        y_pred = torch.cat(all_p)
        val_acc = accuracy(y_true, y_pred)
        val_f1 = macro_f1(y_true, y_pred)
        print(f"[GRU] Epoch {epoch:03d} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            save_path = out_dir / "gru_best.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "config": cfg,
                "scaler": scaler,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, save_path)
            best_path = save_path
            print(f"Saved best GRU model to {save_path}")

        # 每 10 个 epoch 保存一次检查点
        if epoch % 10 == 0:
            ckpt_path = out_dir / f"gru_epoch_{epoch}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "config": cfg,
                "scaler": scaler,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }, ckpt_path)
            print(f"Saved periodic checkpoint: {ckpt_path}")

    # 训练结束，输出最终模型信息（使用当前模型与最佳模型）
    total_params, trainable_params = count_params(model)
    print({
        "final_epoch": int(cfg["num_epochs"]),
        "final_params_total": total_params,
        "final_params_trainable": trainable_params,
        "best_val_acc": best_acc,
        "best_val_f1": best_f1,
        "best_checkpoint": str(best_path) if best_path else None,
    })


if __name__ == "__main__":
    main()
