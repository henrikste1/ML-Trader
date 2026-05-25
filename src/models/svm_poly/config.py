"""

Hyperparameters for the Polynomial Support Vector Regression (SVMP).

"""

DEFAULT_PARAMS: dict = {
    "C":       1.0,
    "epsilon": 0.001,   # 10x smaller than typical daily-return std (~0.013)
    "degree":  3,
    "gamma":   1.0,     # Constant
    "coef0":   1.0,     # Constant
}

# The parameter grid used in rolling-forward validation

PARAM_GRID: dict[str, list] = {
    "C":      [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "degree": [1, 2, 3],
}
