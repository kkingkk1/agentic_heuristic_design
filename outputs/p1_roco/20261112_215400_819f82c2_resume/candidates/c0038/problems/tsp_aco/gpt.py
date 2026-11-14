import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return detour-saving edge indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & off_diagonal

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(finite_values)), 1.0)
    )
    scale = max(scale, np.finfo(float).eps)

    largest = float(np.max(finite_values))
    penalty = max(largest, scale) * 10.0 + scale
    safe = np.where(valid, d, penalty)
    np.fill_diagonal(safe, penalty)

    two_hop = np.min(safe[:, :, None] + safe[None, :, :], axis=1)
    np.fill_diagonal(two_hop, penalty)

    detour_advantage = (two_hop - safe) / scale
    detour_advantage = np.clip(detour_advantage, -30.0, 30.0)

    saving_score = 1.0 / (1.0 + np.exp(-detour_advantage))

    alternative_count = np.sum(valid, axis=1).astype(float)
    endpoint_scarcity = 1.0 / np.sqrt(
        (alternative_count[:, None] + 1.0)
        * (alternative_count[None, :] + 1.0)
    )

    absolute_scale = np.exp(-np.clip(safe / scale, 0.0, 30.0))
    indicators = saving_score * (0.75 + 0.25 * absolute_scale)
    indicators *= 1.0 + 0.5 * endpoint_scarcity

    indicators[~valid] = 0.0
    indicators = np.nan_to_num(
        indicators,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = np.max(indicators)
    if maximum > 0.0:
        indicators /= maximum

    return indicators.astype(float, copy=False)
