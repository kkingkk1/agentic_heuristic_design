import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[finite_positive]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback = max(float(np.max(positive_values)), global_scale)
    replacement = max(fallback * 10.0, eps)

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(3, n - 1)
    nearest = np.partition(work, neighbor_count - 1, axis=1)[:, :neighbor_count]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = work / geometric_scale
    absolute_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_sum = ranks + ranks.T
    rank_score = 1.0 / (1.0 + rank_sum)

    mutual_nearest = np.zeros((n, n), dtype=float)
    nearest_indices = order[:, :neighbor_count]
    rows = np.arange(n)[:, None]
    mutual_nearest[rows, nearest_indices] = 1.0
    mutual_nearest = mutual_nearest * mutual_nearest.T

    density_log = np.log(
        np.maximum(global_scale / local_scale, eps)
    )
    density_gap = np.abs(
        density_log[:, None] - density_log[None, :]
    )
    density_gap /= 1.0 + np.abs(
        density_log[:, None] + density_log[None, :]
    )

    short_edge_score = np.exp(
        -0.85 * np.minimum(normalized_length, 50.0)
    )
    rank_log_score = np.log(np.maximum(rank_score, eps))

    log_score = (
        0.95 * np.log(np.maximum(short_edge_score, eps))
        + 0.85 * rank_log_score
        + 0.38 * mutual_nearest
        - 0.16 * np.minimum(absolute_length, 50.0)
        - 0.24 * density_gap
    )

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
