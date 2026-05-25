"""

Technical indicators and lagged features.

"""

from __future__ import annotations

import numpy as np
import pandas as pd



# Returns

def add_returns(df: pd.DataFrame, horizons=(1, 5, 10, 20)) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for h in horizons:
        out[f"ret_{h}"] = close.pct_change(h)
    out["log_ret_1"] = np.log(close / close.shift(1))
    return out


# Moving averages and ratios

def add_moving_averages(df: pd.DataFrame, windows=(5, 10, 20, 50)) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for w in windows:
        out[f"sma_{w}"] = close.rolling(w).mean()
        out[f"sma_ratio_{w}"] = close / out[f"sma_{w}"] - 1.0
    return out


# RSI (Wilder smoothing)

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out[f"rsi_{window}"] = 100.0 - 100.0 / (1.0 + rs)
    return out


# MACD

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


# Stochastic oscillator

def add_stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    out = df.copy()
    low_k = out["low"].rolling(k).min()
    high_k = out["high"].rolling(k).max()
    out[f"stoch_k_{k}"] = 100.0 * (out["close"] - low_k) / (high_k - low_k).replace(0.0, np.nan)
    out[f"stoch_d_{d}"] = out[f"stoch_k_{k}"].rolling(d).mean()
    return out


# Williams %R

def add_williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    high_w = out["high"].rolling(window).max()
    low_w = out["low"].rolling(window).min()
    out[f"willr_{window}"] = -100.0 * (high_w - out["close"]) / (high_w - low_w).replace(0.0, np.nan)
    return out


# ATR

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out[f"atr_{window}"] = tr.ewm(alpha=1.0 / window, adjust=False).mean()
    return out


# Realised volatility

def add_volatility(df: pd.DataFrame, windows=(10, 20)) -> pd.DataFrame:
    out = df.copy()
    log_ret = np.log(out["close"] / out["close"].shift(1))
    for w in windows:
        out[f"vol_{w}"] = log_ret.rolling(w).std()
    return out

PAPER_FEATURE_COLUMNS = [
    "sma_signal_15",  
    "macd_hist",     
    "stoch_k_14",    
    "stoch_d_3",       
    "stoch_slow_d_3",  
    "rsi_14",
    "willr_14",
]


def build_paper_feature_matrix(
    df: pd.DataFrame,
    sma_window:  int = 15,
    rsi_window:  int = 14,
    wpr_window:  int = 14,
    stoch_k:     int = 14,
    stoch_d:     int = 3,
    macd_fast:   int = 12,
    macd_slow:   int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:

    out = df.copy()
    out = add_macd(out, fast=macd_fast, slow=macd_slow, signal=macd_signal)  
    out = add_stochastic(out, k=stoch_k, d=stoch_d)                           
    slow_d_col = f"stoch_slow_d_{stoch_d}"
    out[slow_d_col] = out[f"stoch_d_{stoch_d}"].rolling(stoch_d).mean()
    out = add_rsi(out, window=rsi_window)                                      
    out = add_williams_r(out, window=wpr_window)                              

    sma = out["close"].rolling(sma_window).mean()
    sma_col = f"sma_signal_{sma_window}"
    out[sma_col] = out["close"] - sma

    feature_cols = [
        sma_col,
        "macd_hist",
        f"stoch_k_{stoch_k}",
        f"stoch_d_{stoch_d}",
        slow_d_col,
        f"rsi_{rsi_window}",
        f"willr_{wpr_window}",
    ]

    # Lag every indicator by 1 day to avoid look-ahead.
    feats = out[feature_cols].shift(1)
    return feats.dropna()
