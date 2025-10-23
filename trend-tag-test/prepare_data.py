from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import qlib  # type: ignore
import yfinance as yf

# 确保 qlib 工具可用
QLIB_ROOT = Path(qlib.__file__).resolve().parents[1]
if str(QLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(QLIB_ROOT))

from scripts.dump_bin import DumpDataAll  # type: ignore


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXPERIMENT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
QLIB_DIR = DATA_ROOT / "qlib"


@dataclass
class DatasetConfig:
    symbols: Tuple[str, ...] = ("QQQ", "TSM")
    start: str = "2013-07-01"  # 覆盖 2014-01-01 的缓冲
    end_exclusive: str = "2025-07-01"  # 覆盖 2024-12-31 的缓冲
    force_refresh: bool = False


def _ensure_dirs(force_refresh: bool) -> None:
    if force_refresh and RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if force_refresh and QLIB_DIR.exists():
        shutil.rmtree(QLIB_DIR)
    QLIB_DIR.mkdir(parents=True, exist_ok=True)


def _extract_single_symbol(df: pd.DataFrame, symbol: str, yf_symbol: str) -> pd.DataFrame:
    if df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df

    for key in (symbol, yf_symbol):
        for level in range(df.columns.nlevels):
            try:
                sub = df.xs(key, axis=1, level=level)
                if isinstance(sub, pd.DataFrame) and not sub.empty:
                    return sub
            except (KeyError, ValueError):
                continue
    raise RuntimeError(f"未能在 yfinance 返回结果中找到 {symbol} 的行情数据列。")


def download_symbol(symbol: str, start: str, end_exclusive: str) -> Path:
    ticker = symbol.replace(".", "-")
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end_exclusive,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False,
        actions=False,
        repair=True,
    )
    if df.empty:
        raise RuntimeError(f"无法下载 {symbol} 的行情数据，请检查网络或参数。")

    df = _extract_single_symbol(df, symbol, ticker)
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})
    elif "date" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "date"})

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df = df[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]]

    csv_path = RAW_DIR / f"{symbol}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def dump_to_qlib() -> None:
    dump = DumpDataAll(
        data_path=str(RAW_DIR),
        qlib_dir=str(QLIB_DIR),
        freq="day",
        date_field_name="date",
        symbol_field_name="symbol",
        exclude_fields="symbol,date",
        max_workers=1,
    )
    dump.dump()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw & qlib data for trend-tag-test experiment.")
    parser.add_argument("--refresh", action="store_true", help="Rebuild dataset (delete cached files).")
    parser.add_argument(
        "--symbols",
        default="QQQ,TSM",
        help="Comma-separated tickers to download (default: QQQ,TSM).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    symbol_list = tuple(sym.strip().upper() for sym in args.symbols.split(",") if sym.strip())
    cfg = DatasetConfig(symbols=symbol_list, force_refresh=args.refresh)

    _ensure_dirs(cfg.force_refresh)

    csv_paths = []
    for symbol in cfg.symbols:
        path = download_symbol(symbol, cfg.start, cfg.end_exclusive)
        csv_paths.append(path)
        print(f"[Data] Saved raw CSV for {symbol}: {path}")

    dump_to_qlib()
    print(f"[Done] qlib dataset ready at {QLIB_DIR}")


if __name__ == "__main__":
    main()
