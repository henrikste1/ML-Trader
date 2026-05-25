"""Data cleaning and time-based train/test splitting."""

from __future__ import annotations

import pandas as pd


# Sort by date, drop duplicates, drop empty rows

def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.dropna(how="all")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out


def train_test_split(
    df: pd.DataFrame, split_date: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:

    split = pd.Timestamp(split_date)
    train = df.loc[df.index < split]
    test = df.loc[df.index >= split]
    return train, test
