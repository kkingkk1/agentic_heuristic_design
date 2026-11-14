import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = 0.5 * (raw + raw.T)
    finite = np.isfinite(d)

    finite_values = d[finite]
    if finite_values.size:
        positive_values = finite_values[finite_values > 0.0]
        fallback = (
            float(np.median(positive_values))
            if positive_values.size
            else max(float(np.max(finite_values)), 1.0)
        )
    else:
        fallback = 1.0

    d = np.nan_to_num(
        d,
        nan=fallback * 10.0,
        posinf=fallback * 10.0,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    np.fill_diagonal(d, np.inf)

    positive = d[off_diagonal & np.isfinite(d) & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else max(fallback, 1.0)
    )
    eps = np.finfo(float).eps * max(global_scale, 1.0)

    finite_d = np.where(np.isfinite(d), d, global_scale * 10.0)
    sorted_d = np.sort(finite_d, axis=1)

    neighborhood_size = max(1, min(n - 1, 4))
    local_scale = np.median(
        sorted_d[:, :neighborhood_size],
        axis=1,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    local_preference = np.exp(-finite_d / pair_scale)
    local_preference[~off_diagonal] = 0.0

    row_sum = local_preference.sum(axis=1, keepdims=True)
    row_preference = local_preference / np.maximum(row_sum, eps)

    mutual_preference = np.sqrt(
        np.maximum(row_preference * row_preference.T, 0.0)
    )

    row_best = np.min(finite_d, axis=1)
    row_median = np.median(finite_d, axis=1)
    row_median = np.maximum(row_median, global_scale * 0.05)

    regret = np.maximum(
        finite_d - row_best[:, None],
        0.0,
    ) / row_median[:, None]

    scarcity = 1.0 / (1.0 + regret)
    scarcity = np.sqrt(np.maximum(scarcity * scarcity.T, 0.0))

    global_preference = np.exp(-finite_d / max(global_scale, eps))
    global_preference[~off_diagonal] = 0.0

    score = (
        0.45 * mutual_preference
        + 0.30 * local_preference
        + 0.15 * scarcity
        + 0.10 * global_preference
    )

    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
