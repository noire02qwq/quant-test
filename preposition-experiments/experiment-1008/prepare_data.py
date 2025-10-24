from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
import yfinance as yf

import qlib

# 将 qlib 根目录加入路径，便于复用 DumpDataAll 工具
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
    """数据集配置：控制标的、时间范围与输出目录。"""

    symbol: str = "TSM"
    start: str = "2014-07-01"  # 起始向前 6 个月，覆盖 2015-01-01
    end_exclusive: str = "2025-07-01"  # 结束向后 6 个月，供 2024-12-31 观察窗口使用
    force_refresh: bool = False


def _ensure_dirs(force_refresh: bool) -> None:
    """确保原始数据与 qlib 输出目录存在，必要时清空刷新。"""

    if force_refresh and RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if force_refresh and QLIB_DIR.exists():
        shutil.rmtree(QLIB_DIR)
    QLIB_DIR.mkdir(parents=True, exist_ok=True)


def _extract_single_symbol(df: pd.DataFrame, symbol: str, yf_symbol: str) -> pd.DataFrame:
    """从 yfinance 下载结果中提取单个标的的数据表。"""

    if df.empty:
        return df

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    for key in (symbol, yf_symbol):
        for level in range(df.columns.nlevels):
            try:
                sub = df.xs(key, axis=1, level=level)
                if isinstance(sub, pd.DataFrame) and not sub.empty:
                    return sub
            except (KeyError, ValueError):
                continue
    raise RuntimeError(f"下载数据中找不到 {symbol} 对应的列，请检查返回格式。")


def download_symbol_data(cfg: DatasetConfig) -> Path:
    """下载指定标的的日线数据并输出为 CSV。"""

    ticker = cfg.symbol.replace(".", "-")
    df = yf.download(
        tickers=ticker,
        start=cfg.start,
        end=cfg.end_exclusive,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False,
        actions=False,
        repair=True,
    )
    if df.empty:
        raise RuntimeError(f"无法下载 {cfg.symbol} 的行情数据，请检查网络或代码。")

    df = _extract_single_symbol(df, cfg.symbol, ticker)
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
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})
    elif "date" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = cfg.symbol
    df = df[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    csv_path = RAW_DIR / f"{cfg.symbol}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def dump_to_qlib() -> None:
    """将 CSV 原始数据转换为 qlib 二进制格式。"""

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
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="准备 preposition-experiments/experiment-1008 的原始及 qlib 数据集")
    parser.add_argument("--refresh", action="store_true", help="重新下载并构建数据（会覆盖现有文件）")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """下载 TSM 行情并输出 qlib 数据集。"""

    args = parse_args(argv)
    cfg = DatasetConfig(force_refresh=args.refresh)
    _ensure_dirs(cfg.force_refresh)
    csv_path = download_symbol_data(cfg)
    dump_to_qlib()
    print(f"数据集已准备完毕：原始 CSV {csv_path}, qlib 目录 {QLIB_DIR}")


if __name__ == "__main__":
    main()
