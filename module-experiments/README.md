# Module Experiments

This directory aggregates the major research sandboxes that used to live at the repository root. Each subfolder is self-contained and can be run independently.

## signal-test
- **Scope**: Exhaustive signal effectiveness evaluation, including indicator computation, triple-barrier labels, and summary tables.
- **Usage**: Run `python module-experiments/signal-test/build_signal_notebook.py` to regenerate the analysis notebook, then execute the notebook (or rely on `jupyter nbconvert --execute`). Outputs (Excel statistics) are stored back under `module-experiments/signal-test`.

## long-classification-experiment
- **Scope**: Transformer-based classification pipeline for long-horizon signals (data prep, training, validation, inference utilities).
- **Usage**: Supply a config JSON (see `configs/`), then run `python module-experiments/long-classification-experiment/train.py --config configs/transformer_raw_traditional.json`. Outputs land under `outputs/` inside the module.

## trend-regression-test
- **Scope**: Regression-focused experiments comparing transformer-family models on trend labels.
- **Usage**: Use `train.py`/`test.py` with the configs in `configs/`. Example:
  ```bash
  python module-experiments/trend-regression-test/train.py --config module-experiments/trend-regression-test/configs/transformer_raw_traditional.json
  ```

## trend-tag-test
- **Scope**: Classification experiments around trend tagging, sharing much of the tooling with the regression testbed but different labels/evaluation.
- **Usage**: Identical interface to the regression module; invoke the desired script with its config under `module-experiments/trend-tag-test/configs/`.

> Add new experiment folders here when needed, together with a concise description and usage snippet.
