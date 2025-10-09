from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests
import yfinance as yf


DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
START_DATE = "2004-01-01"
# yfinance's end date is exclusive; push one day forward to include 2024-12-31
END_DATE_EXCLUSIVE = "2025-01-01"
BATCH_SIZE = 32
HEADERS = {"User-Agent": "Mozilla/5.0"}


def load_sp500_symbols() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    html = response.text
    tables = pd.read_html(StringIO(html), header=0)
    if not tables:
        raise RuntimeError("Failed to parse S&P 500 constituents table")
    df = tables[0]
    symbols = df["Symbol"].dropna().astype(str).str.strip()
    return sorted(symbols.unique())


def yf_symbol(symbol: str) -> str:
    return symbol.replace(".", "-")


def ensure_data_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_batch(symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    yf_symbols = [yf_symbol(sym) for sym in symbols]
    data = yf.download(
        tickers=yf_symbols,
        start=START_DATE,
        end=END_DATE_EXCLUSIVE,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        actions=False,
        repair=True,
    )
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        for original, yf_sym in zip(symbols, yf_symbols):
            if yf_sym not in data.columns.get_level_values(0):
                continue
            ticker_df = data[yf_sym].dropna(how="all")
            if ticker_df.empty:
                continue
            out[original] = ticker_df
    else:
        # Single ticker fallback when only one symbol is requested
        for original, yf_sym in zip(symbols, yf_symbols):
            if not isinstance(data, pd.DataFrame) or data.empty:
                continue
            out[original] = data
    return out


def save_ticker(symbol: str, df: pd.DataFrame, root: Path) -> None:
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df = df.reset_index().rename(columns={"Date": "date"})
    df = df.sort_values("date")
    keep = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    extra = [col for col in df.columns if col not in keep]
    if extra:
        df = df.drop(columns=extra)
    csv_path = root / f"{symbol}.csv"
    df.to_csv(csv_path, index=False)


def main() -> None:
    ensure_data_dir(DATA_DIR)
    symbols = load_sp500_symbols()
    missing: List[str] = []
    for idx in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[idx : idx + BATCH_SIZE]
        print(f"Downloading batch {idx // BATCH_SIZE + 1} / {(len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE}")
        batch_data = download_batch(batch)
        for sym in batch:
            df = batch_data.get(sym)
            if df is None:
                missing.append(sym)
                continue
            save_ticker(sym, df, DATA_DIR)
    if missing:
        missing_path = DATA_DIR / "missing_symbols.txt"
        missing_path.write_text("\n".join(missing))
        print(f"Warning: {len(missing)} symbols missing data. See {missing_path}")
    else:
        print("All symbols downloaded successfully.")


if __name__ == "__main__":
    main()
