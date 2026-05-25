"""

Performance metrics used to evaluate strategies.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TRADING_DAYS = config.TRADING_DAYS_PER_YEAR


# Annualised metrics

def arc(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    cum = (1.0 + r).prod()
    years = len(r) / TRADING_DAYS
    return cum ** (1.0 / years) - 1.0


def cagr(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return float("nan")
    years = len(eq) / TRADING_DAYS
    if years <= 0:
        return float("nan")
    ratio = float(eq.iloc[-1]) / float(eq.iloc[0])
    if ratio <= 0:
        return float("nan")
    return ratio ** (1.0 / years) - 1.0


def asd(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    r = returns.dropna()
    daily_target = (1.0 + target) ** (1.0 / TRADING_DAYS) - 1.0
    downside = (r - daily_target).clip(upper=0.0)
    return float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna()
    if r.std(ddof=1) == 0 or len(r) == 0:
        return float("nan")
    daily_rf = (1.0 + rf) ** (1.0 / TRADING_DAYS) - 1.0
    excess = r - daily_rf
    return float(excess.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def adjusted_sharpe_ratio(returns: pd.Series, equity: pd.Series, rf: float = 0.0) -> float:
    R = cagr(equity)
    s = asd(returns)
    if s == 0 or np.isnan(s) or np.isnan(R):
        return float("nan")
    return max(R / s, 0.0)


def information_ratio(returns: pd.Series) -> float:
    a = arc(returns)
    s = asd(returns)
    if s == 0 or np.isnan(s):
        return float("nan")
    return a / s


def information_ratio_star(returns: pd.Series, equity: pd.Series) -> float:
    R = cagr(equity)
    s = asd(returns)
    mdd = max_drawdown(equity)
    if s == 0 or np.isnan(s) or mdd == 0 or np.isnan(mdd) or np.isnan(R):
        return float("nan")
    return max(R, 0.0) ** 2 / (s * mdd)


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if len(eq) == 0:
        return float("nan")
    running_max = eq.cummax()
    dd = 1.0 - eq / running_max
    return float(dd.max())


def calmar_ratio(returns: pd.Series, equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    R = cagr(equity)
    if mdd == 0 or np.isnan(mdd) or np.isnan(R):
        return float("nan")
    return max(R, 0.0) / mdd


def sortino_ratio(returns: pd.Series, equity: pd.Series, target: float = 0.0) -> float:
    R = cagr(equity)
    sd = downside_deviation(returns, target=target)
    if sd == 0 or np.isnan(sd) or np.isnan(R):
        return float("nan")
    return max(R, 0.0) / sd


def hit_rate(positions: pd.Series, forward_returns: pd.Series) -> float:
    pos = positions.dropna()
    fr = forward_returns.reindex(pos.index)
    pnl = pos * fr
    active = pnl[pos != 0]
    if len(active) == 0:
        return float("nan")
    return float((active > 0).mean())



# One-shot summary

def summary(
    returns: pd.Series,
    equity: pd.Series,
    positions: pd.Series | None = None,
    forward_returns: pd.Series | None = None,
    rf: float = 0.0,
) -> dict[str, float]:
    out = {
        "CAGR":     cagr(equity),
        "ARC":      arc(returns),
        "ASD":      asd(returns),
        "Sharpe":   sharpe_ratio(returns, rf=rf),
        "aSharpe":  adjusted_sharpe_ratio(returns, equity, rf=rf),
        "IR":       information_ratio(returns),
        "IR_star":  information_ratio_star(returns, equity),
        "MDD":      max_drawdown(equity),
        "Calmar":   calmar_ratio(returns, equity),
        "Sortino":  sortino_ratio(returns, equity),
    }
    if positions is not None and forward_returns is not None:
        out["HitRate"] = hit_rate(positions, forward_returns)
    return out
