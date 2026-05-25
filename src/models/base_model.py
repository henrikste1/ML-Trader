"""

Abstract base class shared by every model.

"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseModel":
        ...

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        return self.predict_score(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.0) -> pd.Series:
        score = self.predict_score(X)
        return (score > threshold).astype(int)


    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        with open(path, "rb") as f:
            return pickle.load(f)
