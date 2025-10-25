from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import torch

from data_pipeline import apply_scaler, build_dataloaders, load_scaler, prepare_feature_dataframe, split_by_date
from models import create_model
from train import evaluate, EXPERIMENT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model for long-classification-experiment")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (defaults to best model)")
    parser.add_argument("--split", default="test", help="Split name or 'all'")
    parser.add_argument("--run-dir", default=None, help="Override run/output directory")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    config = load_json(Path(args.config))
    run_dir = Path(args.run_dir or config.get("output_dir", EXPERIMENT_ROOT / "outputs" / "training"))
    metadata_path = run_dir / f"metadata_{config.get('model', {}).get('name', 'transformer')}_{config.get('feature_mode', 'raw_traditional')}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到 metadata.json，请确认训练流程已完成：{metadata_path}")
    metadata = load_json(metadata_path)

    feature_mode = metadata["feature_mode"]
    feature_cols = metadata["feature_cols"]
    window_size = int(metadata["window_size"])
    splits = metadata.get("splits", config.get("splits", {}))

    frame = prepare_feature_dataframe(feature_mode)
    frames = split_by_date(frame, splits)

    scaler = load_scaler(Path(metadata["scaler_path"]))
    scaled_frames = {name: apply_scaler(df, scaler) for name, df in frames.items()}

    loaders = build_dataloaders(
        scaled_frames,
        feature_cols,
        window_size,
        batch_size=int(metadata.get("batch_size", config.get("batch_size", 64))),
        num_workers=int(config.get("num_workers", 0)),
    )

    model_name = metadata["model_name"]
    model_params = metadata.get("model_params", {})
    model = create_model(model_name, input_dim=len(feature_cols), seq_len=window_size, **model_params)

    checkpoint_path = Path(args.checkpoint or metadata.get("best_checkpoint", ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"指定或默认的 checkpoint 不存在：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    device_name = args.device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model.to(device)

    target_splits: Sequence[str]
    if args.split.lower() == "all":
        target_splits = list(loaders.keys())
    else:
        if args.split not in loaders:
            raise KeyError(f"未知数据切分：{args.split}，可选 {list(loaders.keys())}")
        target_splits = [args.split]

    for split_name in target_splits:
        loader = loaders[split_name]
        loss, acc = evaluate(model, loader, device)
        print(f"[{split_name}] loss={loss:.4f} acc={acc:.4f}")


if __name__ == "__main__":
    main()
