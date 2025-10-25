# preposition-experiments

This directory consolidates the independent research prototypes previously scattered under `experiment-*`. Each subfolder is self-contained; details and usage guidance are summarized below.

## experiment-1001 — Qlib US Dataset Builder
- **Goal**: Build a Qlib-compatible daily dataset for a selected US universe using Yahoo Finance.
- **Key scripts**: `build_us_dataset.py`, `datashow.py`.
- **Usage**:
  ```bash
  python build_us_dataset.py --start 2010-01-01 --end 2025-08-31 --force-refresh
  ```
  The script now expects outputs under `data/qlib_us_selected` relative to repo root, so paths remain unchanged after relocation.

## experiment-1002 — Traditional Indicators & Trade Labels
- **Goal**: Compute classic technical indicators, secondary signals, and triple-barrier labels for research datasets.
- **Key scripts**: `technical_indicators.py`, `secondary_signals.py`, `trade_labels.py`.
- **Usage**:
  ```bash
  python technical_indicators.py
  python secondary_signals.py --symbol TSM --start 2024-01-01 --end 2024-12-31
  ```
  Outputs and data loading paths have been updated to reflect the new directory structure.

## experiment-1003 — Sequence Models for Signal Classification
- **Goal**: Train/test GRU and custom models on signal classification tasks.
- **Key scripts**: `train.py`, `train_gru.py`, `validate.py`, `test.py`.
- **Usage**:
  ```bash
  python train.py --config config.json
  python validate.py --config config.json
  ```
Generated artifacts are stored in `preposition-experiments/experiment-1003/outputs`.

## experiment-1005 — Multi-Model Indicator Pipeline
- **Goal**: Provide data preparation, indicator computation, and modelling utilities for multiple strategies.
- **Key scripts**: `ensure_dataset.py`, `indicator_pipeline.py`, `train.py`.
- **Usage**:
  ```bash
  python ensure_dataset.py
  python train.py --config configs/baseline.yaml
  ```
  Data paths now default to `preposition-experiments/experiment-1005/data`.

## experiment-1008 — Advanced Data Pipeline with Label Generation
- **Goal**: Prepare features, compute labels, and train models on extended experiments.
- **Key scripts**: `prepare_data.py`, `compute_labels.py`, `train.py`, `infer_and_plot.py`.
- **Usage**:
  ```bash
  python prepare_data.py
  python compute_labels.py
  python train.py --config configs/default.yaml
  ```
  Scripts save intermediate artifacts in the local `data/` and `outputs/` subdirectories under this experiment.

## experiment-1010 — Variant Pipeline & Modelling Experiments
- **Goal**: Explore alternative labelling rules and model configurations.
- **Key scripts**: `prepare_data.py`, `compute_labels.py`, `train.py`, `infer_and_plot.py`.
- **Usage**:
  ```bash
  python prepare_data.py --config configs/variant.yaml
  python compute_labels.py --config configs/variant.yaml
  python train.py --config configs/variant.yaml
  ```
  Storage conventions align with experiment-1008 (local `data/` & `outputs/`).

---

> **Note**: All code now references the relocated paths under `preposition-experiments/`. If you introduce new experiments, add a similar summary entry here.
