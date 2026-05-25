"""
Per market, model configuration search.

Usage:
    python scripts/configuration_search.py --model svml --markets DAX FTSE100
    python scripts/configuration_search.py --model svmp --markets all
    python scripts/configuration_search.py --model bglm --workers 3
    python scripts/configuration_search.py --model svml --workers 5

"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Type

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import config, data_loader, preprocessor, feature_engineering, labeling
from src.common import walk_forward, backtester, metrics
from src.common.tuning import TunedModel
from src.strategies.ml_strategy import quantile_positions
from src.models.base_model import BaseModel
from src.models.svm.model import LinearSVMModel
from src.models.svm.config import PARAM_GRID as SVML_GRID
from src.models.svm_poly.model import PolynomialSVMModel
from src.models.svm_poly.config import PARAM_GRID as SVMP_GRID
from src.models.bayesian_glm.model import BayesianGLMModel
from src.models.bayesian_glm.config import PARAM_GRID as BGLM_GRID


@dataclass(frozen=True)
class ModelSpec:
    model_cls: Type[BaseModel]
    param_grid: dict
    output_name: str
    inner_cv_overrides: dict | None = None
    final_overrides: dict | None = None


MODELS: dict[str, ModelSpec] = {
    "svml": ModelSpec(LinearSVMModel,     SVML_GRID, "linear_svm"),
    "svmp": ModelSpec(PolynomialSVMModel, SVMP_GRID, "svm_poly"),
    "bglm": ModelSpec(
        BayesianGLMModel,
        BGLM_GRID,
        "bayesian_glm",
        final_overrides={"n_advi": 5_000, "n_posterior": 500}, # Time saver
    ),
}


CONFIGURATIONS: list[tuple[str, dict, float, float, str]] = [
    ("base",                    {}, 0.40, 0.60, "long_short"), # Default

    ("q_30_70_ls",              {}, 0.30, 0.70, "long_short"),
    ("q_35_65_ls",              {}, 0.35, 0.65, "long_short"),
    ("q_45_55_ls",              {}, 0.45, 0.55, "long_short"),

    ("long_only_40_60",         {}, 0.40, 0.60, "long_only"),
    ("long_only_30_70",         {}, 0.30, 0.70, "long_only"),
    ("long_only_median",        {}, 0.50, 0.50, "long_only"),

    # Feature parameters
    ("sma_14",     {"sma_window":  14}, 0.40, 0.60, "long_short"),
    ("sma_16",     {"sma_window":  16}, 0.40, 0.60, "long_short"),
    ("rsi_13",     {"rsi_window":  13}, 0.40, 0.60, "long_short"),
    ("rsi_15",     {"rsi_window":  15}, 0.40, 0.60, "long_short"),
    ("wpr_13",     {"wpr_window":  13}, 0.40, 0.60, "long_short"),
    ("wpr_15",     {"wpr_window":  15}, 0.40, 0.60, "long_short"),
    ("stoch_13",   {"stoch_k":     13}, 0.40, 0.60, "long_short"),
    ("stoch_15",   {"stoch_k":     15}, 0.40, 0.60, "long_short"),

    ("macd_short", {"macd_fast": 11, "macd_slow": 25, "macd_signal":  8}, 0.40, 0.60, "long_short"),
    ("macd_long",  {"macd_fast": 13, "macd_slow": 27, "macd_signal": 10}, 0.40, 0.60, "long_short"),
]


def _grid_combinations(param_grid: dict) -> list[dict]:
    import itertools

    keys = list(param_grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*(param_grid[k] for k in keys))]


def make_model(spec: ModelSpec) -> BaseModel:

    combos = _grid_combinations(spec.param_grid)
    if len(combos) == 1:
        params = {**combos[0], **(spec.final_overrides or {})}
        return spec.model_cls(params=params)
    return TunedModel(
        spec.model_cls,
        param_grid=spec.param_grid,
        val_size=config.TEST_WINDOW,
        scoring="neg_mse",
        inner_cv_overrides=spec.inner_cv_overrides,
        final_overrides=spec.final_overrides,
    )


def run_one_feature_set(
    market: str,
    model_key: str,
    feature_kwargs: dict,
    grouped_configs: list[tuple[str, dict, float, float, str]],
) -> tuple[str, str, list[dict]]:
    np.random.seed(config.RANDOM_SEED)

    spec = MODELS[model_key]
    raw = preprocessor.clean(data_loader.download_index(market))
    X = feature_engineering.build_paper_feature_matrix(raw, **feature_kwargs)
    y = labeling.next_day_return(raw, horizon=1)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]

    trim = walk_forward.trim_to_full_folds(
        X.index, train_window=config.TRAIN_WINDOW, step=config.STEP,
    )
    Xt, yt = X.loc[trim], y.loc[trim]
    fwd = yt

    oos_score = []
    for tr, te in walk_forward.rolling_splits(
        index=Xt.index,
        train_window=config.TRAIN_WINDOW,
        test_window=config.TEST_WINDOW,
        step=config.STEP,
    ):
        m = make_model(spec)
        m.fit(Xt.loc[tr], yt.loc[tr])
        oos_score.append(m.predict_score(Xt.loc[te]))
    oos_score = pd.concat(oos_score).sort_index()

    fwd_oos = fwd.loc[oos_score.index]
    bh = backtester.buy_and_hold_backtest(fwd_oos, cost_bps=config.TRANSACTION_COST_BPS)
    bh_metrics = metrics.summary(
        bh["net_return"],
        bh["equity"],
        pd.Series(1.0, index=fwd_oos.index),
        fwd_oos,
    )

    rows: list[dict] = []
    for config_name, _, q_low, q_high, mode in grouped_configs:
        pos = quantile_positions(oos_score, q_low=q_low, q_high=q_high, mode=mode)
        bt = backtester.backtest(pos, fwd_oos, cost_bps=config.TRANSACTION_COST_BPS)
        ml_metrics = metrics.summary(bt["net_return"], bt["equity"], pos, fwd_oos)
        rows.append({
            "config":  config_name,
            "q_low":   q_low,
            "q_high":  q_high,
            "mode":    mode,
            **{f"feat_{k}": v for k, v in feature_kwargs.items()},
            "n_long":  int((pos > 0).sum()),
            "n_flat":  int((pos == 0).sum()),
            "n_short": int((pos < 0).sum()),
            **{f"ml_{k}": v for k, v in ml_metrics.items()},
            **{f"bh_{k}": v for k, v in bh_metrics.items()},
        })
    return market, model_key, rows


def _runner(args):
    market, model_key, feature_kwargs, grouped_configs = args
    feature_name = ", ".join(f"{k}={v}" for k, v in feature_kwargs.items()) or "base_features"
    t0 = time.time()
    try:
        _, _, rows = run_one_feature_set(market, model_key, feature_kwargs, grouped_configs)
        names = ",".join(r["config"] for r in rows)
        return market, model_key, feature_name, names, rows, None, time.time() - t0
    except Exception as exc:
        return market, model_key, feature_name, "", None, repr(exc), time.time() - t0



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"],
                        default="all",
                        help="Which model(s) to run the search on.")
    parser.add_argument("--markets", nargs="+", default=None,
                        help="Markets to run on. Defaults to every TICKERS entry.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent processes.")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Optional subset of configuration names to run.")
    parser.add_argument("--append-existing", action="store_true",
                        help="Merge rows into an existing configuration CSV instead of replacing it.")
    args = parser.parse_args()

    markets = args.markets or list(config.TICKERS.keys())
    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    if args.configs:
        wanted = set(args.configs)
        configs = [c for c in CONFIGURATIONS if c[0] in wanted]
    else:
        configs = CONFIGURATIONS

    grouped_configs: dict[tuple[tuple[str, object], ...], list[tuple[str, dict, float, float, str]]] = {}
    feature_kwargs_by_key: dict[tuple[tuple[str, object], ...], dict] = {}
    for c in configs:
        key = tuple(sorted(c[1].items()))
        grouped_configs.setdefault(key, []).append(c)
        feature_kwargs_by_key[key] = c[1]

    print(f"Markets : {markets}")
    print(f"Models  : {model_keys}")
    print(f"Configs : {len(configs)}")
    print(f"Feature fits per market/model: {len(grouped_configs)}")
    print(f"Workers : {args.workers}")
    print(f"Total   : {len(markets) * len(model_keys) * len(grouped_configs)} feature-fit runs")
    print()

    rows_by_key: dict[tuple[str, str], list[dict]] = {
        (m, mk): [] for m in markets for mk in model_keys
    }

    work = [
        (m, mk, feature_kwargs_by_key[key], grouped)
        for m in markets
        for mk in model_keys
        for key, grouped in grouped_configs.items()
    ]

    overall_t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_runner, w) for w in work]
        for n, fut in enumerate(as_completed(futures), 1):
            market, model_key, feature_name, names, rows, err, dt = fut.result()
            if err is not None:
                print(f"  [FAIL] {market}/{model_key}/{feature_name} ({dt:.1f}s): {err}")
            else:
                rows_by_key[(market, model_key)].extend(rows)
                best = max(rows, key=lambda r: r.get("ml_IR_star", float("-inf")))
                ml_sharpe = best.get("ml_Sharpe", float("nan"))
                bh_sharpe = best.get("bh_Sharpe", float("nan"))
                print(f"  [{n:>3d}/{len(work)}] {market:>9s}/{model_key}/{feature_name:<34s} "
                      f"({dt:6.1f}s)  rows={len(rows):>2d}  best={best['config']:<18s} "
                      f"Sharpe={ml_sharpe:+.3f}  B&H={bh_sharpe:+.3f}")

    # Persist one CSV per (market, model)
    out_dir = ROOT / "results" / "tables"
    for (market, model_key), rows in rows_by_key.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # Sort: same order as CONFIGURATIONS for readability
        order = {c[0]: i for i, c in enumerate(configs)}
        df["_order"] = df["config"].map(order)
        df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
        csv_name = f"{market}_{MODELS[model_key].output_name}_config_search.csv"
        csv_path = out_dir / csv_name
        if args.append_existing and csv_path.exists():
            existing = pd.read_csv(csv_path)
            existing = existing[~existing["config"].isin(df["config"])]
            df = pd.concat([existing, df], ignore_index=True, sort=False)
            full_order = {c[0]: i for i, c in enumerate(CONFIGURATIONS)}
            df["_order"] = df["config"].map(full_order).fillna(len(full_order))
            df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print(f"\nWrote {csv_path}  ({len(df)} configs)")

    print(f"\nTOTAL wall: {(time.time() - overall_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
