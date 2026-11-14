import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = distances.copy()
    d[~np.isfinite(d)] = np.inf
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result / np.max(result)

    positive_values = finite_values[finite_values > 0]
    fallback = float(np.max(positive_values)) if positive_values.size else 1.0
    global_scale = float(np.median(positive_values)) if positive_values.size else fallback
    global_scale = max(global_scale, np.finfo(float).eps)

    d[~np.isfinite(d)] = fallback * 10.0

    order = np.argsort(d, axis=1)
    sorted_d = np.take_along_axis(d, order, axis=1)

    local_index = min(max(1, n // 4), n - 1)
    local_scale = sorted_d[:, local_index]
    local_scale[~np.isfinite(local_scale)] = global_scale
    local_scale = np.maximum(local_scale, 0.15 * global_scale)

    scale_matrix = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    scale_matrix = np.maximum(scale_matrix, np.finfo(float).eps)

    normalized = d / scale_matrix
    inverse = 1.0 / (1.0 + normalized)
    exponential = np.exp(-np.minimum(normalized, 30.0))

    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(1.5, 0.20 * n))

    nearest = sorted_d[:, 0]
    nearest = np.maximum(nearest, np.finfo(float).eps)
    mutuality = np.sqrt(
        (nearest[:, None] / (d + nearest[:, None]))
        * (nearest[None, :] / (d + nearest[None, :]))
    )

    row_median = np.median(d, axis=1)
    regret = np.maximum(
        0.0,
        d - np.minimum(row_median[:, None], row_median[None, :]),
    ) / (global_scale + scale_matrix)
    regret_gate = 1.0 / (1.0 + regret)

    score = (
        0.30 * inverse
        + 0.25 * exponential
        + 0.20 * rank_score
        + 0.17 * mutuality
        + 0.08 * regret_gate
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    positive = score[score > 0]
    if positive.size == 0:
        score = np.ones((n, n), dtype=float)
        np.fill_diagonal(score, 0.0)
    else:
        score += 0.01 * float(np.median(positive))
        np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
