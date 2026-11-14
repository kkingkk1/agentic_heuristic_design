import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    if not np.any(finite_nonnegative):
        return np.zeros((n, n), dtype=float)

    values = d[finite_nonnegative]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        low = float(np.percentile(positive, 10.0))
        high = float(np.percentile(positive, 90.0))
        median = float(np.median(positive))
    else:
        low = high = median = max(float(np.median(values)), 1.0)

    spread = max(high - low, median * 0.05, eps)

    # Make the prior independent of input direction and mark unusable edges.
    safe = np.where(finite_nonnegative, d, np.inf)
    safe = 0.5 * (safe + safe.T)
    valid = np.isfinite(safe) & ~np.eye(n, dtype=bool)

    finite_safe = safe[valid]
    if finite_safe.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = max(float(np.percentile(finite_safe, 90.0)), median, 1.0)
    working = np.where(valid, safe, fallback + 20.0 * spread)

    # Smooth global preference: no hard nearest-neighbor rank or local product.
    center = float(np.percentile(finite_safe, 35.0))
    temperature = max(0.35 * spread, eps)
    preference = 1.0 / (
        1.0 + np.exp(np.clip((working - center) / temperature, -60.0, 60.0))
    )
    preference[~valid] = 0.0

    # Endpoint competition: normalize each edge by all alternatives at that endpoint.
    row_mass = preference.sum(axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, eps)
    directed = preference / row_mass

    # Harmonic aggregation rewards edges supported by both endpoints.
    reverse = directed.T
    support = (2.0 * directed * reverse) / np.maximum(directed + reverse, eps)

    # Penalize edges that are unusually costly for both of their endpoints.
    endpoint_median = np.median(
        np.where(valid, working, np.nan),
        axis=1,
    )
    endpoint_median = np.nan_to_num(
        endpoint_median,
        nan=fallback,
        posinf=fallback,
        neginf=fallback,
    )
    local_cost = np.sqrt(
        np.maximum(endpoint_median[:, None] * endpoint_median[None, :], eps)
    )
    excess = np.maximum(working - local_cost, 0.0)
    cost_penalty = np.exp(-excess / max(spread, eps))

    score = support * cost_penalty
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
