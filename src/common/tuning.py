"""

Per-fold hyperparameter tuning

"""

from __future__ import annotations

import itertools
from typing import Any, Type

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from ..models.base_model import BaseModel


class TunedModel(BaseModel):
    name: str = "tuned"

    def __init__(
        self,
        model_cls: Type[BaseModel],
        param_grid: dict[str, list],
        val_size: int = 20,
        scoring: str = "neg_mse",
        inner_cv_overrides: dict | None = None,
        final_overrides: dict | None = None,
    ):
        if not param_grid:
            raise ValueError("param_grid must be non-empty.")
        if val_size < 1:
            raise ValueError("val_size must be >= 1.")
        self.model_cls = model_cls
        self.param_grid = param_grid
        self.val_size = int(val_size)
        self.scoring = scoring
        self.inner_cv_overrides = inner_cv_overrides or {}
        self.final_overrides = final_overrides or {}

        self.best_params_: dict | None = None
        self.best_score_: float | None = None
        self.cv_results_: pd.DataFrame | None = None
        self.best_model_: BaseModel | None = None

        base_name = getattr(model_cls, "name", model_cls.__name__)
        self.name = f"tuned_{base_name}"


    def _grid_combinations(self):
        keys = list(self.param_grid.keys())
        for vals in itertools.product(*(self.param_grid[k] for k in keys)):
            yield dict(zip(keys, vals))

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
            try:
                return -float(mean_squared_error(y_true, y_pred))
            except ValueError:
                return float("nan")


    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "TunedModel":
        n = len(X_train)
        if n <= self.val_size:
            raise ValueError(
                f"In-sample size ({n}) must exceed val_size ({self.val_size})."
            )

        train_size = n - self.val_size
        X_tr = X_train.iloc[:train_size]
        y_tr = y_train.iloc[:train_size]
        X_va = X_train.iloc[train_size:]
        y_va = y_train.iloc[train_size:]

        records: list[dict[str, Any]] = []
        for params in self._grid_combinations():
            inner_params = {**params, **self.inner_cv_overrides}
            m = self.model_cls(params=inner_params)
            m.fit(X_tr, y_tr)
            pred = m.predict_score(X_va)
            records.append({**params, "score": self._score(y_va.values, pred.values)})

        self.cv_results_ = (
            pd.DataFrame(records)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        best_row_idx = int(np.nanargmax([r["score"] for r in records]))
        self.best_params_ = {k: records[best_row_idx][k] for k in self.param_grid.keys()}
        self.best_score_ = float(self.cv_results_.iloc[0]["score"])

        # Final fit on the full in-sample window using the best params
        final_params = {**self.best_params_, **self.final_overrides}
        self.best_model_ = self.model_cls(params=final_params)
        self.best_model_.fit(X_train, y_train)
        return self

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        if self.best_model_ is None:
            raise RuntimeError("Model has not been fit yet.")
        return self.best_model_.predict_score(X)
