"""Matplotlib helpers for equity curves, drawdowns and metric comparisons."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    equities: Mapping[str, pd.Series],
    title: str = "Equity curves",
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=figsize)
    for label, eq in equities.items():
        ax.plot(eq.index, eq.values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_drawdown(
    equity: pd.Series,
    title: str = "Drawdown",
    figsize: tuple[float, float] = (10, 3),
) -> plt.Figure:
    
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(dd.index, dd.values, 0.0, color="tab:red", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metric_comparison(
    table: pd.DataFrame,
    metric: str,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=figsize)
    table[metric].plot(kind="bar", ax=ax)
    ax.set_title(title or metric)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
