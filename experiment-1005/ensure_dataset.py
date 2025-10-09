from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
TARGET_START = pd.Timestamp("2010-01-01")
TARGET_END = pd.Timestamp("2024-12-31")
END_EXCLUSIVE = "2025-01-01"


def list_csv_symbols(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("*.csv"))


def coverage_ok(path: Path) -> tuple[bool, pd.Timestamp, pd.Timestamp]:
    df = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
    if df.empty:
        return False, pd.NaT, pd.NaT
    start = df["date"].min()
    end = df["date"].max()
    ok = start <= TARGET_START and end >= TARGET_END
    return ok, start, end


def download_symbol(symbol: str) -> pd.DataFrame:
    print(f"[ensure_dataset] downloading {symbol} ...")
    data = yf.download(
        symbol,
        start=str(TARGET_START.date()),
        end=END_EXCLUSIVE,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(symbol, axis=1, level=0, drop_level=False).droplevel(0, axis=1)
    data = data.reset_index().rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    data = data.sort_values("date")
    return data


def ensure_symbol(path: Path, overwrite: bool) -> bool:
    symbol = path.stem
    ok, start, end = coverage_ok(path)
    if ok:
        print(f"[ensure_dataset] {symbol} ok (start={start.date()}, end={end.date()})")
        return False
    print(
        f"[ensure_dataset] {symbol} coverage insufficient (start={start}, end={end}). Attempting refresh..."
    )
    df = download_symbol(symbol)
    if df.empty:
        print(f"[ensure_dataset] WARNING: no data received for {symbol}; keeping original file")
        return False
    new_start = df["date"].min()
    new_end = df["date"].max()
    if new_start > TARGET_START or new_end < TARGET_END:
        print(
            f"[ensure_dataset] WARNING: refreshed data for {symbol} still insufficient "
            f"(start={new_start.date()}, end={new_end.date()})."
        )
    if overwrite:
        df.to_csv(path, index=False)
        print(f"[ensure_dataset] updated {path.name}")
    return True


def main(symbols: Iterable[str] | None = None, overwrite: bool = True) -> None:
    data_dir = DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = list_csv_symbols(data_dir)
    touched = 0
    for csv_path in csv_files:
        if symbols and csv_path.stem.upper() not in {s.upper() for s in symbols}:
            continue
        if ensure_symbol(csv_path, overwrite=overwrite):
            touched += 1
    print(f"[ensure_dataset] Completed. Updated {touched} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensure dataset coverage from 2010-01-01 to 2024-12-31")
    parser.add_argument("--symbols", nargs="*", help="Optional list of symbols to refresh")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite CSV files")
    args = parser.parse_args()
    main(symbols=args.symbols, overwrite=not args.no_overwrite)
