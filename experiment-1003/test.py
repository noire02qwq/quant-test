"""测试脚本：加载最优模型并在测试集评估。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from data import load_dataframe, prepare_splits, make_loaders
from model import TimeSeriesTransformer
from train import pick_device, accuracy, macro_f1


def main() -> None:
    cfg_path = Path("experiment-1003/config.json")
    cfg: Dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    device = pick_device(str(cfg.get("device", "auto")))

    data_path = Path(cfg["data_path"])
    df = load_dataframe(data_path, index_col=cfg.get("index_col", "datetime"))
    ds_train, ds_val, ds_test, feature_cols = prepare_splits(df, cfg, window_size=int(cfg["window_size"]))
    _, _, test_loader = make_loaders(ds_train, ds_val, ds_test, batch_size=int(cfg["batch_size"]))

    ckpt = torch.load(Path(cfg["output_dir"]) / "models" / "best.pt", map_location="cpu")
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
    print({"test_acc": accuracy(y_true, y_pred), "test_macro_f1": macro_f1(y_true, y_pred)})


if __name__ == "__main__":
    main()

