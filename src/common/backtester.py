"""
Strategy backtester

Converts model predictions into positions

"""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(
    positions: pd.Series,
    forward_returns: pd.Series,
) -> pd.DataFrame:
    
    df = pd.DataFrame(index=positions.index)
    df["position"] = positions
    df["gross_return"] = positions * forward_returns.reindex(positions.index)
    df["equity"] = (1.0 + df["gross_return"].fillna(0.0)).cumprod()
    return df


def buy_and_hold_backtest(
    forward_returns: pd.Series,
) -> pd.DataFrame:
    
    positions = pd.Series(1.0, index=forward_returns.index, name="position")
    return backtest(positions, forward_returns)
