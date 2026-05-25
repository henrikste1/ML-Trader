"""

Data loading utilities.

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config


def _raw_path(market: str) -> Path:
    return config.RAW_DIR / f"{market}.csv"


def _processed_path(market: str) -> Path:
    return config.PROCESSED_DIR / f"{market}.parquet"


def _resolve_ticker(market: str) -> str:
    if market in config.TICKERS:
        return config.TICKERS[market]
    if market in config.EXTERNAL_TICKERS:
        return config.EXTERNAL_TICKERS[market]
    raise KeyError(
        f"No ticker mapping for {market!r}. "
        f"Add it to TICKERS or EXTERNAL_TICKERS in src/common/config.py."
    )


def download_index(
    market: str,
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:

    ticker = ticker or _resolve_ticker(market)
    start = start or config.START_DATE
    end = end or config.END_DATE

    path = _raw_path(market)
    if path.exists() and not force:
        return load_raw(market)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {market} ({ticker}).")

    # Flatten yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df.index.name = "date"
    df.to_csv(path)
    return df


def load_raw(market: str) -> pd.DataFrame:
    path = _raw_path(market)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file for {market} not found at {path}. "
            f"Run scripts/download_data.py first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df.sort_index()


def load_processed(market: str) -> pd.DataFrame:
    path = _processed_path(market)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file for {market} not found at {path}. "
            f"Run the data-prep cell of a notebook or 00_data_exploration.ipynb."
        )
    return pd.read_parquet(path)


def save_processed(market: str, df: pd.DataFrame) -> None:
    df.to_parquet(_processed_path(market))
