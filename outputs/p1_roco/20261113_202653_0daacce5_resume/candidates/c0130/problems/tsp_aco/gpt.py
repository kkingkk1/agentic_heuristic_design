import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structurally informed edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    positive = distances[finite & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).tiny)
    eps = np.sqrt(np.finfo(float).eps) * scale

    d = np.where(valid, np.maximum(distances, 0.0), np.inf)
    np.fill_diagonal(d, np.inf)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    inv = np.where(valid, 1.0 / (d + eps), 0.0)
    row_sum = np.sum(inv, axis=1, keepdims=True)
    col_sum = np.sum(inv, axis=0, keepdims=True)

    # Degree balancing: reduce excessive concentration on already attractive endpoints.
    balance = 1.0 / np.sqrt((row_sum + eps) * (col_sum + eps))
    positive_balance = balance[balance > 0.0]
    balance /= np.median(positive_balance) if positive_balance.size else 1.0
    balance = np.clip(balance, 0.5, 2.0)

    # Temperature-calibrated distance preference.
    inv_scale = np.median(inv[inv > 0.0]) if np.any(inv > 0.0) else 1.0
    inverse_term = inv / (inv + inv_scale + eps)
    thermal_term = np.exp(-np.minimum(d, 50.0 * scale) / (1.5 * scale + eps))
    thermal_term[~valid] = 0.0
    attraction = 0.7 * inverse_term + 0.3 * thermal_term

    # Endpoint regret: preserve edges that are substantially better than alternatives.
    first_out = np.min(d, axis=1)
    first_in = np.min(d, axis=0)

    if n >= 3:
        two_out = np.partition(d, 1, axis=1)[:, 1]
        two_in = np.partition(d, 1, axis=0)[1, :]
    else:
        two_out = np.full(n, np.inf)
        two_in = np.full(n, np.inf)

    regret_out = np.clip(two_out[:, None] - d, 0.0, None) / (scale + eps)
    regret_in = np.clip(two_in[None, :] - d, 0.0, None) / (scale + eps)
    regret = 1.0 + 0.18 * np.tanh(regret_out + regret_in)
    regret[~valid] = 1.0

    # Approximate bypass score using the best alternatives at both endpoints.
    best_out = np.argmin(d, axis=1)
    best_in = np.argmin(d, axis=0)

    alt_out = np.where(
        best_out[:, None] == np.arange(n)[None, :],
        two_out[:, None],
        first_out[:, None],
    )
    alt_in = np.where(
        best_in[None, :] == np.arange(n)[:, None],
        two_in[None, :],
        first_in[None, :],
    )

    bypass = np.minimum(alt_out + alt_in, 50.0 * scale)
    bypass_term = bypass / (bypass + np.maximum(d, 0.0) + scale + eps)
    bypass_term[~valid] = 0.0

    priors = attraction * balance * regret * (0.75 + 0.25 * bypass_term)
    priors[~valid] = 0.0
    np.fill_diagonal(priors, 0.0)
    priors = np.nan_to_num(priors, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = np.max(priors)
    if maximum > 0.0:
        priors /= maximum

    return priors
