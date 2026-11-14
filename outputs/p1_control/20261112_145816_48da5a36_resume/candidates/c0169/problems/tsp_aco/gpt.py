import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0)
    undirected_valid = valid & valid.T

    if not np.any(undirected_valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    fallback = float(np.max(positive_values))
    finite_d = np.where(valid, d, fallback * 10.0)
    finite_d = np.maximum(finite_d, np.finfo(float).eps)

    neighbor_values = np.where(valid, d, np.inf)
    sorted_neighbors = np.sort(neighbor_values, axis=1)
    kth = min(2, n - 2)
    local_scale = sorted_neighbors[:, kth]

    missing_scale = ~np.isfinite(local_scale)
    if np.any(missing_scale):
        local_scale[missing_scale] = np.median(positive_values)

    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = finite_d / geometric_scale

    order = np.argsort(finite_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks /= max(n - 1, 1)

    mutual_rank = 0.5 * (ranks + ranks.T)

    absolute_scale = float(np.median(positive_values))
    absolute_scale = max(absolute_scale, np.finfo(float).eps)

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None:]
    balance_ratio = scale_difference / (
        scale_sum + np.finfo(float).eps
    )

    log_score = (
        -1.18 * relative_length
        -3.45 * mutual_rank
        -0.58 * finite_d / absolute_scale
        -1.72 * balance_ratio
    )

    log_score -= np.max(log_score[undirected_valid])
    score = np.exp(np.clip(log_score, -745.0, 0.0))

    score[~undirected_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
