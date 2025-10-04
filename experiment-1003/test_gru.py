"""测试脚本（GRU）：在测试集评估最优 GRU 模型。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from data import load_dataframe, prepare_frames, apply_scaler, SeriesConfig, WindowDataset, make_loaders
from gru_model import GRUClassifier
from train_gru import pick_device, accuracy, macro_f1


def main() -> None:
    cfg: Dict = json.loads(Path("experiment-1003/config.json").read_text(encoding="utf-8"))
    device = pick_device(str(cfg.get("device", "auto")))

    df = load_dataframe(Path(cfg["data_path"]), index_col=cfg.get("index_col", "datetime"))
    train_df, val_df, test_df, feature_cols = prepare_frames(df, cfg)

    ckpt = torch.load(Path(cfg["output_dir"]) / "models" / "gru_best.pt", map_location="cpu")
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

    model = GRUClassifier(feature_dim=len(feature_cols))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

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
    print({"test_acc": accuracy(y_true, y_pred), "test_macro_f1": macro_f1(y_true, y_pred)})


if __name__ == "__main__":
    main()
