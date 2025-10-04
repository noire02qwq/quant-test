"""测试脚本：加载最优模型并在测试集评估。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from data import load_dataframe, prepare_frames, apply_scaler, SeriesConfig, WindowDataset, make_loaders
from model import TimeSeriesTransformer
from train import pick_device, accuracy, macro_f1


def main() -> None:
    cfg_path = Path("experiment-1003/config.json")
    cfg: Dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    device = pick_device(str(cfg.get("device", "auto")))

    data_path = Path(cfg["data_path"])
    df = load_dataframe(data_path, index_col=cfg.get("index_col", "datetime"))
    train_df, val_df, test_df, feature_cols = prepare_frames(df, cfg)

    ckpt = torch.load(Path(cfg["output_dir"]) / "models" / "best.pt", map_location="cpu")
    scaler = ckpt.get("scaler", None)
    if scaler is not None:
        train_df = apply_scaler(train_df, scaler)
        val_df = apply_scaler(val_df, scaler)
        test_df = apply_scaler(test_df, scaler)
    series_cfg = SeriesConfig(window_size=int(cfg["window_size"]), feature_cols=feature_cols)
    ds_train = WindowDataset(train_df, series_cfg)
    ds_val = WindowDataset(val_df, series_cfg)
    ds_test = WindowDataset(test_df, series_cfg)
    _, _, test_loader = make_loaders(ds_train, ds_val, ds_test, batch_size=int(cfg["batch_size"]))

    model_cfg = cfg["model"]
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
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_y, all_p = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            pred = logits.argmax(dim=1)
            all_y.append(y.cpu())
            all_p.append(pred.cpu())
    y_true = torch.cat(all_y)
    y_pred = torch.cat(all_p)
    print({"test_acc": accuracy(y_true, y_pred), "test_macro_f1": macro_f1(y_true, y_pred, num_classes=num_classes)})


if __name__ == "__main__":
    main()
