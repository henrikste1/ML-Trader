"""

Bayesian Generalised Linear Model — regression form (Gaussian likelihood).

"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import MinMaxScaler

from ...common import config
from ..base_model import BaseModel
from .config import DEFAULT_PARAMS


class BayesianGLMModel(BaseModel):
    name = "bayesian_glm"

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.scaler: MinMaxScaler | None = None
        self._feature_columns: list[str] | None = None
        self._beta_samples: np.ndarray | None = None     
        self._intercept_samples: np.ndarray | None = None   

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BayesianGLMModel":
        self._feature_columns = list(X_train.columns)

        self.scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        X = self.scaler.fit_transform(X_train.values)
        y = y_train.values.astype(float)

        n_features = X.shape[1]
        prior_sigma     = float(self.params["prior_sigma"])
        intercept_sigma = float(self.params["intercept_sigma"])
        noise_sigma     = float(self.params["noise_sigma"])
        n_advi          = int(self.params["n_advi"])
        n_posterior     = int(self.params["n_posterior"])

        with pm.Model():
            intercept = pm.Normal("intercept", mu=0.0, sigma=intercept_sigma)
            beta      = pm.Normal("beta",      mu=0.0, sigma=prior_sigma, shape=n_features)
            sigma     = pm.HalfNormal("sigma",  sigma=noise_sigma)
            mu        = intercept + pm.math.dot(X, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            approx = pm.fit(
                n=n_advi,
                method="advi",
                progressbar=False,
                random_seed=config.RANDOM_SEED,
            )
            trace = approx.sample(draws=n_posterior, random_seed=config.RANDOM_SEED)

        self._beta_samples = trace.posterior["beta"].values.reshape(-1, n_features)
        self._intercept_samples = trace.posterior["intercept"].values.reshape(-1)
        return self

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        if self._beta_samples is None or self.scaler is None:
            raise RuntimeError("Model has not been fit yet.")
        X = X[self._feature_columns]
        Xs = self.scaler.transform(X.values)

        mu = self._intercept_samples[None, :] + Xs @ self._beta_samples.T
        pred_return = mu.mean(axis=1)
        return pd.Series(pred_return, index=X.index, name="pred_return")
