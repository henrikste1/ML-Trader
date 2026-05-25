"""Passive Buy-and-Hold benchmark."""

from __future__ import annotations

import pandas as pd

# Constant +1 position aligned to index.

def buy_and_hold_positions(index: pd.Index) -> pd.Series:
    return pd.Series(1.0, index=index, name="position")
