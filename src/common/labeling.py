"""Supervised-learning target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

def next_day_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:

    close = df["close"]
    r = close.shift(-horizon) / close - 1.0
    r = r.dropna()
    r.name = "fwd_return"
    return r
