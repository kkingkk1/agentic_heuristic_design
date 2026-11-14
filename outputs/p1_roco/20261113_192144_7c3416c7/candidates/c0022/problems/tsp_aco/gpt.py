import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances, dtype=float)

    d = distances.copy()
    finite = np.isfinite(d)

    if not np.all(finite):
        finite_values = d[finite]
        replacement = (
            np.max(finite_values) * 10.0
            if finite_values.size
            else 1.0
        )
        d[~finite] = replacement

    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, 0.0)

    if n == 1:
        return np.zeros_like(d)

    off_diagonal = ~np.eye(n, dtype=bool)
    positive = d[off_diagonal & (d > 0)]

    scale = np.median(positive) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    row_median = np.median(
        np.where(off_diagonal, d, np.nan),
        axis=1
    )
    row_median = np.nan_to_num(row_median, nan=scale)
    row_median = np.maximum(row_median, np.finfo(float).eps)

    local_spread = np.median(
        np.abs(d - row_median[:, None])[off_diagonal]
    )
    temperature = max(float(local_spread), 0.05 * scale, np.finfo(float).eps)

    preference = np.exp(
        -np.minimum(d / temperature, 60.0)
    )
    np.fill_diagonal(preference, 0.0)

    row_sum = preference.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, np.finfo(float).eps)
    row_preference = preference / row_sum

    mutuality = np.sqrt(
        row_preference * row_preference.T
    )

    row_best = np.min(
        np.where(off_diagonal, d, np.inf),
        axis=1
    )
    regret = np.maximum(d - row_best[:, None], 0.0) / row_median[:, None]
    scarcity = 1.0 / (1.0 + regret)
    scarcity = np.sqrt(scarcity * scarcity.T)

    distance_score = np.exp(
        -np.minimum(d / scale, 60.0)
    )

    score = (
        0.45 * distance_score
        + 0.35 * mutuality
        + 0.20 * scarcity
    )

    score *= np.sqrt(
        np.maximum(row_preference, 0.0)
        * np.maximum(row_preference.T, 0.0)
    )

    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
