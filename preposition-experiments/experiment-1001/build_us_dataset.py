"""Utility to build a Qlib-ready daily dataset for selected US tickers.

The script relies solely on qlib's Yahoo collector pipeline to download raw data
and qlib's dumping utilities to generate the binary features/instruments files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import qlib
from loguru import logger

# Ensure qlib's repository root (which contains the "scripts" helpers) is importable.
QLIB_ROOT = Path(qlib.__file__).resolve().parents[1]
if str(QLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(QLIB_ROOT))

from scripts.data_collector import utils as dc_utils  # type: ignore
from scripts.data_collector.yahoo import collector as yahoo_collector  # type: ignore
from scripts.dump_bin import DumpDataAll  # type: ignore


@dataclass
class DatasetConfig:
    """Configuration for the Qlib dataset build."""

    start_date: str = "2010-01-01"
    end_date: str | None = None
    data_root: Path = Path("data/qlib_us_selected")
    max_workers: int = 4
    request_delay: float = 0.2
    force_refresh: bool = False

    sector_map: Dict[str, Sequence[str]] = field(
        default_factory=lambda: {
            "tech": (
                "NVDA",
                "TSM",
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "SAP",
                "ANET",
                "AVGO",
                "IBM",
                "TSLA",
                "PLTR",
                "ADP",
                "ALAB",
                "ADI",
                "TXN",
            ),
            "financial": (
                "JPM",
                "MS",
                "KKR",
                "MAIN",
                "V",
                "AXP",
                "PGR",
                "ICE",
                "BN",
                "SPGI",
                "BX",
                "NDAQ",
                "ARES",
            ),
            "industrial": (
                "ABBV",
                "CAT",
                "RTX",
                "VST",
                "MNST",
                "MCD",
                "CVX",
            ),
        }
    )

    def __post_init__(self) -> None:
        self.start_ts = pd.Timestamp(self.start_date)
        self.end_ts = pd.Timestamp(self.end_date) if self.end_date else pd.Timestamp(datetime.utcnow().date())
        if self.end_ts < self.start_ts:
            raise ValueError("end_date must be greater than or equal to start_date")
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.source_dir = self.data_root / "source"
        self.normalize_dir = self.data_root / "normalize"
        self.qlib_dir = self.data_root / "qlib_data"
        self.symbols: List[str] = sorted({sym.upper() for symbols in self.sector_map.values() for sym in symbols})

    @property
    def start_str(self) -> str:
        return self.start_ts.strftime("%Y-%m-%d")

    @property
    def end_str(self) -> str:
        return self.end_ts.strftime("%Y-%m-%d")


def _empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _patch_symbol_sources(symbols: Sequence[str]) -> None:
    """Monkey patch qlib's Yahoo collector to use the provided symbol list."""

    symbol_list = [sym.upper() for sym in symbols]

    def symbol_provider(_: str | Path | None = None) -> List[str]:
        return list(symbol_list)

    dc_utils.get_us_stock_symbols = symbol_provider  # type: ignore
    if hasattr(dc_utils, "_US_SYMBOLS"):
        dc_utils._US_SYMBOLS = list(symbol_list)  # type: ignore[attr-defined]
    yahoo_collector.get_us_stock_symbols = symbol_provider  # type: ignore

    def custom_get_instrument_list(self) -> List[str]:  # type: ignore[override]
        logger.info("Using custom instrument list with %d symbols", len(symbol_list))
        return list(symbol_list)

    yahoo_collector.YahooCollectorUS.get_instrument_list = custom_get_instrument_list  # type: ignore
    sys.modules["collector"] = yahoo_collector  # required by BaseRun


def _collect_raw_csv(cfg: DatasetConfig) -> None:
    logger.info("Downloading Yahoo daily data for %d symbols", len(cfg.symbols))
    run = yahoo_collector.Run(
        source_dir=str(cfg.source_dir),
        normalize_dir=str(cfg.normalize_dir),
        max_workers=cfg.max_workers,
        interval="1d",
        region="US",
    )
    run.download_data(start=cfg.start_str, end=cfg.end_str, delay=cfg.request_delay)


def _dump_to_qlib(cfg: DatasetConfig) -> None:
    logger.info("Dumping CSV files from %s into Qlib format", cfg.source_dir)
    dump = DumpDataAll(
        data_path=str(cfg.source_dir),
        qlib_dir=str(cfg.qlib_dir),
        freq="day",
        max_workers=cfg.max_workers,
        date_field_name="date",
        symbol_field_name="symbol",
        exclude_fields="symbol,date",
    )
    dump.dump()


def _write_sector_lists(cfg: DatasetConfig) -> None:
    instruments_file = cfg.qlib_dir / "instruments" / "all.txt"
    if not instruments_file.exists():
        raise FileNotFoundError(f"Instrument file not found: {instruments_file}")

    all_df = pd.read_csv(instruments_file, sep="\t", names=["symbol", "start", "end"])
    for name, members in cfg.sector_map.items():
        subset = all_df[all_df["symbol"].isin(sym.upper() for sym in members)].sort_values("symbol")
        if subset.empty:
            logger.warning("No instruments found for sector %s; skipping", name)
            continue
        out_file = instruments_file.parent / f"{name}.txt"
        subset.to_csv(out_file, sep="\t", index=False, header=False)
        logger.info("Wrote %s with %d entries", out_file, len(subset))


def build_dataset(cfg: DatasetConfig) -> None:
    logger.info("Building dataset into %s", cfg.qlib_dir)
    if cfg.force_refresh:
        for path in (cfg.source_dir, cfg.normalize_dir, cfg.qlib_dir):
            _empty_dir(path)
    else:
        for path in (cfg.source_dir, cfg.normalize_dir, cfg.qlib_dir):
            path.mkdir(parents=True, exist_ok=True)

    _patch_symbol_sources(cfg.symbols)
    _collect_raw_csv(cfg)
    _dump_to_qlib(cfg)
    _write_sector_lists(cfg)
    logger.info("Dataset build complete. Qlib data directory: %s", cfg.qlib_dir)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qlib dataset for selected US tickers")
    parser.add_argument("--start", dest="start", default="2010-01-01", help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", dest="end", default=None, help="Inclusive end date (YYYY-MM-DD); defaults to today")
    parser.add_argument(
        "--data-root",
        dest="data_root",
        default="data/qlib_us_selected",
        help="Directory where raw CSVs and Qlib data will be written",
    )
    parser.add_argument("--max-workers", dest="max_workers", type=int, default=4, help="Parallel workers for download/dump")
    parser.add_argument("--delay", dest="request_delay", type=float, default=0.2, help="Pause between Yahoo requests")
    parser.add_argument(
        "--force-refresh",
        dest="force_refresh",
        action="store_true",
        help="Remove existing output directories before rebuilding",
    )
    return parser.parse_args(args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    cfg = DatasetConfig(
        start_date=args.start,
        end_date=args.end,
        data_root=Path(args.data_root),
        max_workers=args.max_workers,
        request_delay=args.request_delay,
        force_refresh=args.force_refresh,
    )
    build_dataset(cfg)


if __name__ == "__main__":
    main()
