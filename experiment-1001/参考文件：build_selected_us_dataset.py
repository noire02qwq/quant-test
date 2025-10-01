from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import fire
import pandas as pd
from loguru import logger
from yahooquery import Ticker

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dump_bin import DumpDataAll


@dataclass
class SelectedUSDatasetBuilder:
    """Download selected US tickers from Yahoo and convert to Qlib format."""

    start_date: str = "2010-01-01"
    end_date: str = "2025-10-01"
    csv_dir: str = "~/.qlib/stock_data/selected_us_daily/csv"
    qlib_dir: str = "~/.qlib/qlib_data/selected_us_daily"
    max_workers: int = 8

    TECH_TICKERS: Sequence[str] = field(
        default_factory=lambda: (
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
        )
    )
    FIN_TICKERS: Sequence[str] = field(
        default_factory=lambda: (
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
        )
    )
    IND_TICKERS: Sequence[str] = field(
        default_factory=lambda: (
            "ABBV",
            "CAT",
            "RTX",
            "VST",
            "MNST",
            "MCD",
            "CVX",
        )
    )

    def __post_init__(self) -> None:
        self.csv_dir = Path(self.csv_dir).expanduser().resolve()
        self.qlib_dir = Path(self.qlib_dir).expanduser().resolve()
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.qlib_dir.mkdir(parents=True, exist_ok=True)
        self.start_ts = pd.Timestamp(self.start_date)
        self.end_ts = pd.Timestamp(self.end_date)
        if self.end_ts < self.start_ts:
            raise ValueError("end_date must be greater than or equal to start_date")
        self.ticker_groups: Dict[str, Sequence[str]] = {
            "tech": self.TECH_TICKERS,
            "financial": self.FIN_TICKERS,
            "industrial": self.IND_TICKERS,
        }
        self.all_tickers: List[str] = sorted(set(itertools.chain.from_iterable(self.ticker_groups.values())))
        logger.info(
            "SelectedUSDatasetBuilder initialized with {} tickers from {} to {}",
            len(self.all_tickers),
            self.start_ts.date(),
            self.end_ts.date(),
        )

    # --- public entry points -------------------------------------------------
    def build(self) -> None:
        """End-to-end build: download CSV, dump to Qlib, create sector files."""
        self.download_csv()
        self.dump_to_qlib()
        self.write_sector_instruments()

    def download_csv(self) -> None:
        """Download Yahoo daily data into per-ticker CSV files."""
        logger.info("Downloading Yahoo data for {} tickers", len(self.all_tickers))
        history = self._fetch_history(self.all_tickers)
        if history.empty:
            raise RuntimeError("Yahoo query returned no data")

        history["date"] = history["date"].apply(self._normalize_timestamp)
        history.sort_values(["symbol", "date"], inplace=True)

        for symbol, df_symbol in history.groupby("symbol", group_keys=False):
            outfile = self.csv_dir.joinpath(f"{symbol}.csv")
            cleaned = self._prepare_symbol_frame(df_symbol)
            cleaned.to_csv(outfile, index=False)
            logger.info("Saved {} with {} rows", outfile, len(cleaned))

    def dump_to_qlib(self) -> None:
        """Convert CSV directory into Qlib binary format."""
        logger.info("Dumping CSV data into Qlib format at {}", self.qlib_dir)
        dump = DumpDataAll(
            data_path=str(self.csv_dir),
            qlib_dir=str(self.qlib_dir),
            freq="day",
            max_workers=self.max_workers,
            date_field_name="date",
            symbol_field_name="symbol",
            exclude_fields="symbol,date",
        )
        dump.dump()
        logger.info("Dump finished")

    def write_sector_instruments(self) -> None:
        """Create sector instrument lists based on build output."""
        instruments_dir = self.qlib_dir.joinpath("instruments")
        all_file = instruments_dir.joinpath("all.txt")
        if not all_file.exists():
            raise FileNotFoundError(f"Instrument file not found: {all_file}")

        all_df = pd.read_csv(all_file, sep="\t", names=["symbol", "start", "end"])
        for name, tickers in self.ticker_groups.items():
            subset = all_df[all_df["symbol"].isin(tickers)].sort_values("symbol")
            if subset.empty:
                logger.warning("No instruments found for group {}", name)
                continue
            out_file = instruments_dir.joinpath(f"{name}.txt")
            subset.to_csv(out_file, sep="\t", index=False, header=False)
            logger.info("Wrote {} with {} entries", out_file, len(subset))

    # --- helpers -------------------------------------------------------------
    def _fetch_history(self, tickers: Sequence[str]) -> pd.DataFrame:
        yahoo_start = self.start_ts.strftime("%Y-%m-%d")
        yahoo_end = (self.end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        ticker_str = " ".join(sorted(set(tickers)))
        ticker = Ticker(ticker_str, asynchronous=True)
        df = ticker.history(start=yahoo_start, end=yahoo_end, interval="1d")
        if isinstance(df, pd.DataFrame):
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            else:
                df = df.copy()
        else:
            df = pd.DataFrame()
        if not df.empty and "symbol" not in df.columns:
            df["symbol"] = df.get("ticker")
        if "symbol" not in df.columns or "date" not in df.columns:
            raise RuntimeError("Yahoo history response missing symbol/date columns")
        missing = set(tickers) - set(df["symbol"].unique())
        if missing:
            logger.warning("No data returned for tickers: {}", ", ".join(sorted(missing)))
        # Filter out rows without essential OHLCV
        essential = ["open", "high", "low", "close", "volume"]
        df = df.dropna(subset=[c for c in essential if c in df.columns], how="all")
        return df

    def _prepare_symbol_frame(self, df_symbol: pd.DataFrame) -> pd.DataFrame:
        df_symbol = df_symbol.copy()
        df_symbol["symbol"] = df_symbol["symbol"].str.upper()
        df_symbol = df_symbol[[
            col
            for col in [
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjclose",
                "volume",
            ]
            if col in df_symbol.columns
        ]]
        df_symbol.rename(columns={"adjclose": "adj_close"}, inplace=True)
        df_symbol["date"] = df_symbol["date"].apply(self._normalize_timestamp)
        df_symbol.dropna(subset=["close"], inplace=True)
        df_symbol = df_symbol[df_symbol["date"].between(self.start_ts, self.end_ts)]
        df_symbol.sort_values("date", inplace=True)
        return df_symbol

    @staticmethod
    def _normalize_timestamp(value) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None and ts.tzinfo.utcoffset(ts) is not None:
            ts = ts.tz_convert(None)
        return ts.normalize()


def main():
    fire.Fire({
        "build": lambda **kwargs: SelectedUSDatasetBuilder(**kwargs).build(),
        "download": lambda **kwargs: SelectedUSDatasetBuilder(**kwargs).download_csv(),
        "dump": lambda **kwargs: SelectedUSDatasetBuilder(**kwargs).dump_to_qlib(),
        "write_sectors": lambda **kwargs: SelectedUSDatasetBuilder(**kwargs).write_sector_instruments(),
    })


if __name__ == "__main__":
    main()
