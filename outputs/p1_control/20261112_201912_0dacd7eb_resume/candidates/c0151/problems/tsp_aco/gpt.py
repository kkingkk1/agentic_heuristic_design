import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    nonnegative = np.maximum(np.where(finite, distances, 0.0), 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    usable = finite & off_diagonal
    positive = nonnegative[usable & (nonnegative > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, nonnegative, fallback)
    d = 0.5 * (d + d.T)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    scale = max(float(np.median(positive)), eps)

    if n == 2:
        value = d[0, 1]
        if not np.isfinite(value) or value <= 0.0:
            return np.zeros((n, n), dtype=float)
        result = np.zeros((n, n), dtype=float)
        result[0, 1] = result[1, 0] = 1.0
        return result

    ordered = np.sort(d, axis=1)

    first = ordered[:, 0]
    second = ordered[:, 1]
    third = ordered[:, 2] if n > 3 else second

    alternatives = 0.5 * (second + third)
    alternatives = np.maximum(alternatives, eps)

    endpoint_budget = alternatives[:, None] + alternatives[None, :]
    direct_cost = d

    savings = endpoint_budget - direct_cost
    savings_fraction = savings / (endpoint_budget + scale + eps)
    savings_fraction = np.clip(savings_fraction, 0.0, 1.0)

    endpoint_gap_i = (alternatives[:, None] - direct_cost) / (
        alternatives[:, None] + scale + eps
    )
    endpoint_gap_j = (alternatives[None, :] - direct_cost) / (
        alternatives[None, :] + scale + eps
    )

    endpoint_gap = 0.5 * (
        np.clip(endpoint_gap_i, 0.0, 1.0)
        + np.clip(endpoint_gap_j, 0.0, 1.0)
    )

    local_scale = 0.5 * (first + alternatives)
    local_scale = np.maximum(local_scale, eps)

    scale_ratio = np.minimum(
        local_scale[:, None] / local_scale[None, :],
        local_scale[None, :] / local_scale[:, None],
    )
    scale_balance = 0.75 + 0.25 * np.sqrt(
        np.clip(scale_ratio, 0.0, 1.0)
    )

    endpoint_pressure = np.sqrt(
        np.clip(
            alternatives[:, None] / (d + alternatives[:, None] + eps),
            0.0,
            1.0,
        )
        * np.clip(
            alternatives[None, :] / (d + alternatives[None, :] + eps),
            0.0,
            1.0,
        )
    )

    score = (
        savings_fraction
        * (0.85 + 0.15 * endpoint_gap)
        * scale_balance
        * (0.80 + 0.20 * endpoint_pressure)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
