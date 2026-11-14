import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    off_diagonal = ~np.eye(n, dtype=bool)

    finite_offdiag = finite & off_diagonal
    if not np.any(finite_offdiag):
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (distances + distances.T)
    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = np.maximum(d, 0.0)

    finite_values = d[np.isfinite(d) & off_diagonal]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif finite_values.size:
        scale = float(np.median(finite_values))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    safe = d.copy()
    safe[~np.isfinite(safe)] = scale * 1e6
    np.fill_diagonal(safe, 0.0)

    valid = np.isfinite(d) & off_diagonal

    row_values = np.where(valid, safe, np.nan)
    row_median = np.nanmedian(row_values, axis=1)
    row_median = np.nan_to_num(row_median, nan=scale, posinf=scale, neginf=scale)
    row_median = np.maximum(row_median, np.finfo(float).eps)

    row_min = np.min(np.where(valid, safe, np.inf), axis=1)
    row_min = np.where(np.isfinite(row_min), row_min, scale)

    endpoint_pressure = np.maximum(
        0.0,
        1.0 - safe / (row_median[:, None] + scale),
    )
    endpoint_pressure = np.sqrt(
        np.maximum(endpoint_pressure, 0.0)
        * np.maximum(endpoint_pressure.T, 0.0)
    )

    distance_attraction = np.exp(
        -np.minimum(safe / scale, 40.0)
    )

    two_hop = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        through_k = safe[:, k, None] + safe[None, k, :]
        through_k[k, :] = np.inf
        through_k[:, k] = np.inf
        two_hop = np.minimum(two_hop, through_k)

    closure_denominator = np.maximum(two_hop, scale)
    closure_advantage = (
        (two_hop - safe) / (closure_denominator + scale)
    )
    closure_advantage = np.clip(closure_advantage, 0.0, 1.0)
    closure_advantage = 0.5 * (
        closure_advantage + closure_advantage.T
    )

    endpoint_gain = np.maximum(
        0.0,
        (np.minimum(row_median[:, None], row_median[None, :]) - safe)
        / (row_median[:, None] + row_median[None, :] + scale),
    )
    endpoint_gain = np.clip(endpoint_gain, 0.0, 1.0)

    score = (
        distance_attraction
        * (
            0.40
            + 0.25 * closure_advantage
            + 0.20 * endpoint_pressure
            + 0.15 * endpoint_gain
        )
    )

    score = 0.5 * (score + score.T)
    score[~(valid | valid.T)] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.clip(score, 0.0, 1.0)
