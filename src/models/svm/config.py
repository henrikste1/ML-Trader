"""

Hyperparameters for the Linear Support Vector Regression (SVML).

"""

DEFAULT_PARAMS: dict = {
    "C":       1.0,
    "epsilon": 0.001,   # 10x smaller than typical daily-return std (~0.013)
}

# The parameter grid used in rolling-forward validation

PARAM_GRID: dict[str, list] = {
    "C": [0.1, 0.5, 1.0, 5.0, 10.0],
}
