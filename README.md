# Quant Test

This repository contains utilities and experiments for quantitative research built on top of [Microsoft Qlib](https://github.com/microsoft/qlib).

## Contents
- `experiment-1001/build_us_dataset.py` – build a custom daily OHLCV dataset for a curated list of US tickers using Qlib's Yahoo collector pipeline. All raw CSV, normalized, and binary data are written under `data/qlib_us_selected/`.
- `experiment-1001/build_us_dataset.md` – usage guide for the dataset builder, covering arguments, directory layout, and refresh tips.
- `experiment-1001/datashow.py` – render TSM candlestick and volume charts between two dates using the locally built Qlib dataset.
- `draft.md` – research planning notes.

## Quick Start
1. **Build the dataset**
   ```bash
   python experiment-1001/build_us_dataset.py --start 2010-01-01 --end 2025-09-29 --force-refresh
   ```
2. **Visualize TSM**
   ```bash
   python experiment-1001/datashow.py
   ```
   (Set `MPLBACKEND=Agg` if running headless.)

## Repository Layout
```
quant-test/
├─ data/qlib_us_selected/    # Generated dataset (created by build_us_dataset.py)
├─ experiment-1001/
│  ├─ build_us_dataset.py
│  ├─ build_us_dataset.md
│  └─ datashow.py
└─ draft.md
```

## Requirements
- Python 3.12+
- [Qlib](https://github.com/microsoft/qlib) installed in the environment.
- Dependencies for Qlib's Yahoo collector (`requests`, `pandas`, `yahooquery`, etc.).

## License
MIT License.
