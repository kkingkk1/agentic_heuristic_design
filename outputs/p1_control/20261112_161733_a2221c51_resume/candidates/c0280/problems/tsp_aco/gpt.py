import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
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

    symmetric_distance = np.zeros_like(d, dtype=float)
    symmetric_distance[valid] = 0.5 * (d[valid] + d.T[valid])

    edge_values = symmetric_distance[valid]
    global_scale = max(float(np.median(edge_values)), eps)
    fallback_distance = max(float(np.max(edge_values)), global_scale)

    safe_distance = np.where(
        valid,
        symmetric_distance,
        fallback_distance * 10.0,
    )
    np.fill_diagonal(safe_distance, fallback_distance * 10.0)

    neighborhood_rank = min(1, n - 2)
    local_scale = np.partition(
        safe_distance,
        kth=neighborhood_rank,
        axis=1,
    )[:, neighborhood_rank]

    finite_local = np.isfinite(local_scale) & (local_scale > 0.0)
    local_scale = np.where(
        finite_local,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = safe_distance / endpoint_scale
    global_length = safe_distance / global_scale

    order = np.argsort(
        safe_distance,
        axis=1,
        kind="stable",
    )

    ranks = np.empty((n, n), dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]
    ranks = ranks / max(n - 1, 1)
    ranks = np.where(valid, ranks, 1.0)

    mutual_rank = ranks + ranks.T
    rank_preference = np.exp(-3.0 * mutual_rank)

    endpoint_preference = np.exp(-1.10 * relative_length)

    raw_global_preference = np.exp(
        -0.26 * global_length
    )
    global_preference = 0.68 + 0.32 * raw_global_preference

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    scale_balance = 1.0 / (
        1.0 + 0.85 * scale_difference / (scale_sum + eps)
    )
    neighborhood_balance = 0.78 + 0.22 * scale_balance

    excess_longness = np.maximum(
        relative_length - 1.0,
        0.0,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            np.log(np.maximum(endpoint_preference, eps))
            + np.log(np.maximum(rank_preference, eps))
            + np.log(np.maximum(global_preference, eps))
            + np.log(np.maximum(neighborhood_balance, eps))
            - 0.12 * excess_longness
        )

    finite_scores = valid & np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))

    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(
            log_score[finite_scores] - maximum,
            -745.0,
            0.0,
        )
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
