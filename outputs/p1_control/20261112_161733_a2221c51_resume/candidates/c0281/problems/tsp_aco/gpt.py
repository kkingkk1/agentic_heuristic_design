import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority scores for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = np.zeros_like(d, dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    positive_values = symmetric_d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    replacement = max(10.0 * fallback_distance, global_scale)
    safe = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(safe, replacement)

    neighbor_index = min(1, n - 2)
    local_scale = np.partition(
        safe,
        kth=neighbor_index,
        axis=1,
    )[:, neighbor_index]
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = safe / np.maximum(endpoint_scale, eps)
    global_length = safe / global_scale

    endpoint_preference = endpoint_scale / np.maximum(safe, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)
    ranks[~valid] = 1.0

    mutual_rank = np.sqrt(
        np.maximum(1.0 - ranks, eps)
        * np.maximum(1.0 - ranks.T, eps)
    )

    log_scale_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.35 * np.minimum(log_scale_gap, 50.0)
    )

    endpoint_longness = np.maximum(
        safe / local_scale[:, None],
        safe / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)

    global_preference = 0.68 + 0.32 * np.exp(
        -0.26 * np.minimum(global_length, 1e8)
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            0.90 * np.log(np.maximum(endpoint_preference, eps))
            + 1.15 * np.log(np.maximum(mutual_rank, eps))
            + 0.45 * np.log(np.maximum(scale_compatibility, eps))
            + np.log(np.maximum(global_preference, eps))
            - 0.55 * np.minimum(relative_length, 1e8)
            - 0.22 * np.minimum(excess_longness, 1e8)
        )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
