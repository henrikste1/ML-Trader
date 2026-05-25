"""
Download all configured indices into data/raw/.

Usage:
    python scripts/download_data.py            # only downloads missing data
    python scripts/download_data.py --force    # re-download everything

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import config                 # noqa: E402
from src.common.data_loader import download_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download cached files.")
    args = parser.parse_args()

    for market, ticker in config.TICKERS.items():
        print(f"[{market}] downloading {ticker} ({config.START_DATE} -> {config.END_DATE}) ...")
        df = download_index(market, ticker=ticker, force=args.force)
        print(f"[{market}] {len(df):,} rows written to data/raw/{market}.csv")

if __name__ == "__main__":
    main()
