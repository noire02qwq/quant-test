from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = EXPERIMENT_ROOT / "train.py"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs" / "transformer_alpha.json"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs" / "tuning_runs"


def frange(start: float, stop: float, step: float) -> Iterable[float]:
    value = start
    while value <= stop + 1e-9:
        yield value
        value += step


ALPHA_VARIANTS = ["alpha158", "alpha360"]
WINDOW_SIZES = list(range(80, 161, 16))
BATCH_SIZES = list(range(128, 257, 16))
LEARNING_RATES = [3e-4, 4e-4, 5e-4, 6e-4]
DMODELS = [128, 256, 384, 512]
FEEDFORWARD = list(range(256, 769, 128))
NUM_LAYERS = [8, 16]
DROPOUTS = [round(x, 2) for x in frange(0.30, 0.50, 0.05)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search trainer for the transformer regressor.")
    parser.add_argument("--base-config", default=str(DEFAULT_CONFIG), help="Base config JSON path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for tuning runs.")
    parser.add_argument("--max-configs", type=int, default=None, help="Max number of configs to run (useful for smoke tests).")
    parser.add_argument("--dry-run", action="store_true", help="Only generate config files without launching training.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose metadata already exists.")
    return parser.parse_args()


def load_base_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_grid() -> Iterable[Mapping[str, object]]:
    cartesian = itertools.product(
        ALPHA_VARIANTS,
        WINDOW_SIZES,
        BATCH_SIZES,
        LEARNING_RATES,
        DMODELS,
        FEEDFORWARD,
        NUM_LAYERS,
        DROPOUTS,
    )
    for alpha_kind, window, batch, lr, d_model, dim_ff, num_layers, dropout in cartesian:
        yield {
            "alpha_kind": alpha_kind,
            "window_size": window,
            "batch_size": batch,
            "learning_rate": lr,
            "d_model": d_model,
            "dim_feedforward": dim_ff,
            "num_layers": num_layers,
            "dropout": dropout,
        }


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_training(config_path: Path) -> None:
    cmd = ["python", str(TRAIN_SCRIPT.name), "--config", str(config_path)]
    subprocess.run(cmd, check=True, cwd=EXPERIMENT_ROOT)


def build_result_row(
    run_name: str,
    alpha_kind: str,
    combo: Mapping[str, object],
    metadata_path: Path,
) -> Dict[str, object]:
    row = {
        "run": run_name,
        "alpha_kind": alpha_kind,
        "window_size": combo["window_size"],
        "batch_size": combo["batch_size"],
        "learning_rate": combo["learning_rate"],
        "d_model": combo["d_model"],
        "dim_feedforward": combo["dim_feedforward"],
        "num_layers": combo["num_layers"],
        "dropout": combo["dropout"],
        "best_epoch": None,
        "best_val_mae": None,
        "best_val_mse": None,
        "test_mae": None,
        "test_mse": None,
    }
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row.update(
            {
                "best_epoch": metadata.get("best_epoch"),
                "best_val_mae": metadata.get("best_val_mae"),
                "best_val_mse": metadata.get("best_val_mse"),
                "test_mae": metadata.get("test_mae"),
                "test_mse": metadata.get("test_mse"),
            }
        )
    return row


def main() -> None:
    args = parse_args()
    base_config = load_base_config(Path(args.base_config))
    output_root = Path(args.output_root)
    ensure_output_dir(output_root)

    results: List[Dict[str, object]] = []
    counters = {variant: 0 for variant in ALPHA_VARIANTS}
    total_processed = 0

    for combo in build_grid():
        alpha_kind = combo["alpha_kind"]  # type: ignore[index]
        counters.setdefault(alpha_kind, 0)
        counters[alpha_kind] += 1
        seq = counters[alpha_kind]
        tag = "158" if alpha_kind.endswith("158") else "360"
        run_name = f"transformer_{tag}_{seq:03d}"
        run_dir = output_root / run_name
        ensure_output_dir(run_dir)

        config = deepcopy(base_config)
        config.setdefault("feature_mode", "alpha")
        feature_params = dict(config.get("feature_mode_params", {}))
        feature_params["alpha_kind"] = alpha_kind
        feature_params.setdefault("min_r2", 0.75)
        config["feature_mode_params"] = feature_params
        config["window_size"] = combo["window_size"]  # type: ignore[index]
        config["batch_size"] = combo["batch_size"]  # type: ignore[index]
        config["learning_rate"] = combo["learning_rate"]  # type: ignore[index]
        config.setdefault("model", {}).setdefault("params", {})
        model_params = config["model"]["params"]
        model_params["d_model"] = combo["d_model"]  # type: ignore[index]
        model_params["dim_feedforward"] = combo["dim_feedforward"]  # type: ignore[index]
        model_params["num_layers"] = combo["num_layers"]  # type: ignore[index]
        model_params["dropout"] = combo["dropout"]  # type: ignore[index]
        config["output_dir"] = str(run_dir)

        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        metadata_name = f"metadata_{config['model']['name']}_{config['feature_mode']}.json"
        metadata_path = run_dir / metadata_name
        if args.skip_existing and metadata_path.exists():
            print(f"[Skip] {run_name} already completed.")
            total_processed += 1
            results.append(build_result_row(run_name, alpha_kind, combo, metadata_path))
            if args.max_configs and total_processed >= args.max_configs:
                break
            continue

        if not args.dry_run:
            print(f"[Run] {run_name} -> {config_path}")
            run_training(config_path)
        else:
            print(f"[DryRun] Prepared {run_name} -> {config_path}")

        results.append(build_result_row(run_name, alpha_kind, combo, metadata_path))

        total_processed += 1
        if args.max_configs and total_processed >= args.max_configs:
            break

    if results:
        df = pd.DataFrame(results)
        excel_path = output_root / "tuning_summary.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"[Summary] Saved {len(results)} rows to {excel_path}")
    else:
        print("[Summary] No runs executed.")


if __name__ == "__main__":
    main()
