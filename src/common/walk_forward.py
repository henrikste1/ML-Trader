"""

Walk-forward cross-validation.

"""

from __future__ import annotations

from typing import Iterator

import pandas as pd


def trim_to_full_folds(
    index: pd.Index,
    train_window: int,
    step: int,
) -> pd.Index:

    n = len(index)
    if n < train_window + step:
        raise ValueError(
            f"Need at least train_window + step = {train_window + step} rows, got {n}."
        )
    excess = (n - train_window) % step
    return index[excess:]


def rolling_splits(
    index: pd.Index,
    train_window: int,
    test_window: int,
    step: int | None = None,
) -> Iterator[tuple[pd.Index, pd.Index]]:

    n = len(index)
    step = step or test_window

    start = 0
    while start + train_window + test_window <= n:
        train_idx = index[start:start + train_window]
        test_idx = index[start + train_window:start + train_window + test_window]
        yield train_idx, test_idx
        start += step
