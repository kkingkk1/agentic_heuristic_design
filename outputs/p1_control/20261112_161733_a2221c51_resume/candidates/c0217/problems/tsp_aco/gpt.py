import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise scores for stochastic TSP sampling."""
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

    symmetric_d = 0.5 * (d + d.T)
    positive_values = symmetric_d[valid]

    global_scale = max(float(np.median(positive_values)), eps)
    largest = max(float(np.max(positive_values)), global_scale)
    replacement = max(10.0 * largest, global_scale)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    # Local scale: the second-nearest valid edge when available.
    neighbor_index = min(1, n - 2)
    local_scale = np.partition(work, neighbor_index, axis=1)[:, neighbor_index]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = work / endpoint_scale
    global_length = work / global_scale

    # Soft bilateral endpoint preference.
    endpoint_affinity = endpoint_scale / np.maximum(work, eps)

    # Continuous ranks measured independently by both endpoints.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks = ranks / max(n - 1, 1)
    ranks[~valid] = 1.0

    bilateral_rank = np.sqrt(
        np.maximum(1.0 - ranks, eps)
        * np.maximum(1.0 - ranks.T, eps)
    )

    # Comparable local scales indicate compatible neighborhoods.
    log_scale_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(-0.35 * np.minimum(log_scale_gap, 50.0))

    # Penalize only edges that are unusually long for at least one endpoint.
    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            0.85 * np.log(np.maximum(endpoint_affinity, eps))
            + 1.10 * np.log(np.maximum(bilateral_rank, eps))
            + 0.45 * np.log(np.maximum(scale_compatibility, eps))
            - 0.65 * relative_length
            - 0.12 * global_length
            - 0.24 * excess_longness
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
