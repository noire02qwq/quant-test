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

from data import load_dataframe, prepare_splits, make_loaders
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

    # 加载数据并切分
    df = load_dataframe(data_path, index_col=cfg.get("index_col", "datetime"))
    ds_train, ds_val, ds_test, feature_cols = prepare_splits(df, cfg, window_size=int(cfg["window_size"]))
    train_loader, val_loader, _ = make_loaders(ds_train, ds_val, ds_test, batch_size=int(cfg["batch_size"]))

    # 构建模型
    model_cfg = cfg["model"]
    model = TimeSeriesTransformer(
        feature_dim=len(feature_cols),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        dropout=float(model_cfg["dropout"]),
        pooling=str(model_cfg.get("pooling", "mean")),
        num_classes=3,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
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
        val_f1 = macro_f1(y_true, y_pred)

        print(f"Epoch {epoch:03d} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        # 保存最优
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = out_dir / "best.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "config": cfg,
            }, save_path)
            print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    main()
