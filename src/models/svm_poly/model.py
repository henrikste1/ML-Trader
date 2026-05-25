"""

Polynomial Support Vector Regression (SVMP).

"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from ..base_model import BaseModel
from .config import DEFAULT_PARAMS


class PolynomialSVMModel(BaseModel):
    name = "svm_poly"

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.pipeline: Pipeline | None = None
        self._feature_columns: list[str] | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "PolynomialSVMModel":
        self._feature_columns = list(X_train.columns)
        self.pipeline = Pipeline([
            ("scaler", MinMaxScaler(feature_range=(-1.0, 1.0))),
            ("svr", SVR(kernel="poly", **self.params)),
        ])
        self.pipeline.fit(X_train.values, y_train.values)
        return self

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fit yet.")
        X = X[self._feature_columns]
        score = self.pipeline.predict(X.values)
        return pd.Series(score, index=X.index, name="pred_return")
