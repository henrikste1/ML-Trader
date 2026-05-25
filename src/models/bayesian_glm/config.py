"""

Hyperparameters for the Bayesian GLM regression model.

"""

DEFAULT_PARAMS: dict = {
    "prior_sigma":     0.005,
    "intercept_sigma": 0.05,
    "noise_sigma":     0.02,
    "n_advi":          5_000,
    "n_posterior":     500,
}

# The parameter grid used in rolling-forward validation

PARAM_GRID: dict[str, list] = {
    "prior_sigma": [0.005],
}
