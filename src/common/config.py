"""

Global project configurations.

"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR:      Path = PROJECT_ROOT / "data"
RAW_DIR:       Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
RESULTS_DIR:   Path = PROJECT_ROOT / "results"
FIGURES_DIR:   Path = RESULTS_DIR / "figures"
TABLES_DIR:    Path = RESULTS_DIR / "tables"
MODELS_DIR:    Path = RESULTS_DIR / "models"

for _p in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

TICKERS: dict[str, str] = {
    "OSEBX":     "OSEBX.OL",
    "FTSE100":   "^FTSE",
    "Nikkei225": "^N225",
}

ACTIVE_MARKETS: list[str] = list(TICKERS.keys())

END_DATE:   str = date.today().isoformat()
START_DATE: str = (date.today() - timedelta(days=365 * 20)).isoformat()

TRAIN_WINDOW: int = 200
TEST_WINDOW:  int = 20
STEP:         int = 20

RANDOM_SEED: int = 42

TRADING_DAYS_PER_YEAR: int = 252
