"""训练脚本：读取配置，构建数据与模型，执行训练与验证并保存最优模型。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import (
    load_dataframe,
    prepare_frames,
    compute_scaler,
    apply_scaler,
    SeriesConfig,
    WindowDataset,
    make_loaders,
)
from model import TimeSeriesTransformer


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


def main() -> None:
    # 读取配置
    cfg_path = Path("experiment-1003/config.json")
    cfg: Dict = json.loads(cfg_path.read_text(encoding="utf-8"))

    set_seed(int(cfg["random_seed"]))
    device = pick_device(str(cfg.get("device", "auto")))

    data_path = Path(cfg["data_path"])
    out_dir = Path(cfg["output_dir"]) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据并切分（先构建特征与标签，再基于训练集标准化）
    df = load_dataframe(data_path, index_col=cfg.get("index_col", "datetime"))
    train_df, val_df, test_df, feature_cols = prepare_frames(df, cfg)
    scaler = compute_scaler(train_df, feature_cols)
    train_df = apply_scaler(train_df, scaler)
    val_df = apply_scaler(val_df, scaler)
    test_df = apply_scaler(test_df, scaler)
    series_cfg = SeriesConfig(window_size=int(cfg["window_size"]), feature_cols=feature_cols)
    ds_train = WindowDataset(train_df, series_cfg)
    ds_val = WindowDataset(val_df, series_cfg)
    ds_test = WindowDataset(test_df, series_cfg)
    train_loader, val_loader, _ = make_loaders(ds_train, ds_val, ds_test, batch_size=int(cfg["batch_size"]))

    # 构建模型
    model_cfg = cfg["model"]
    # 根据 label_mode 选择类别数
    label_mode = str(cfg.get("label_mode", "binary")).lower()
    num_classes = 2 if label_mode == "binary" else 3
    model = TimeSeriesTransformer(
        feature_dim=len(feature_cols),
        d_model=int(model_cfg.get("d_model", 128)),
        nhead=int(model_cfg.get("nhead", 4)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pooling=str(model_cfg.get("pooling", "mean")),
        num_classes=num_classes,
        norm_first=bool(model_cfg.get("norm_first", True)),
        embedding_dropout=float(model_cfg.get("embedding_dropout", 0.05)),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    # 类别不平衡：使用训练集分布计算权重（确保 1D int64 非负，二分类）
    labels = ds_train.y
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    labels = labels.view(-1).long().clamp_min(0)
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_counts = class_counts.clamp(min=1.0)
    weights = (class_counts.sum() / (class_counts.numel() * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1 = -1.0
    best_acc = -1.0
    best_path = None
    num_epochs = int(cfg["num_epochs"])
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            # 防止梯度爆炸导致 NaN
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
        val_f1 = macro_f1(y_true, y_pred, num_classes=num_classes)

        print(f"Epoch {epoch:03d} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        # 保存最优
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            save_path = out_dir / "best.pt"
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
            print(f"Saved best model to {save_path}")

        # 每 10 个 epoch 保存一次检查点
        if epoch % 10 == 0:
            ckpt_path = out_dir / f"transformer_epoch_{epoch}.pt"
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

    # 训练结束统计打印
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({
        "final_epoch": num_epochs,
        "final_params_total": total_params,
        "final_params_trainable": trainable_params,
        "best_val_acc": best_acc,
        "best_val_f1": best_f1,
        "best_checkpoint": str(best_path) if best_path else None,
    })


if __name__ == "__main__":
    main()
