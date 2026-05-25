"""

Convert model scores to trading positions.

"""

from __future__ import annotations

import pandas as pd

from ..common import config

def quantile_positions(
    score: pd.Series,
    q_low: float = 0.40,
    q_high: float = 0.60,
    mode: str = "long_short",
) -> pd.Series:

    s = score.dropna()
    q_lo = s.quantile(q_low)
    q_hi = s.quantile(q_high)

    pos = pd.Series(0.0, index=s.index, name="position")
    pos[s >= q_hi] = 1.0
    pos[s <= q_lo] = -1.0
    if mode == "long_only":
        pos = pos.clip(lower=0.0)
    elif mode != "long_short":
        raise ValueError(f"Unknown strategy mode: {mode!r}")
    return pos
