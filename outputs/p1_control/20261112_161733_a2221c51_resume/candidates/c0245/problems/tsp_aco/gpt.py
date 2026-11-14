import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
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
    maximum_distance = max(float(np.max(positive_values)), global_scale)
    replacement = max(10.0 * maximum_distance, global_scale)

    work = np.where(valid, symmetric_d, replacement)
    np.fill_diagonal(work, replacement)

    # Use a moderately local neighborhood scale to reduce nearest-neighbor noise.
    neighborhood_rank = min(2, n - 2)
    local_scale = np.partition(
        work, neighborhood_rank, axis=1
    )[:, neighborhood_rank]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = work / endpoint_scale
    global_length = work / global_scale

    # Geometric mean of the two endpoint potentials.
    endpoint_potential = np.sqrt(
        (local_scale[:, None] / np.maximum(work, eps))
        * (local_scale[None, :] / np.maximum(work, eps))
    )
    endpoint_potential = np.maximum(endpoint_potential, eps)

    # Stable ranks measure whether both endpoints independently prefer an edge.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks[~valid] = float(n)

    rank_affinity = np.exp(
        -0.5 * (
            ranks / max(n - 1, 1)
            + ranks.T / max(n - 1, 1)
        )
    )
    rank_affinity = np.maximum(rank_affinity, eps)

    # Prefer edges connecting nodes with comparable local scales.
    scale_mismatch = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.40 * np.minimum(scale_mismatch, 50.0)
    )

    # Apply a soft penalty only when the edge is longer than either endpoint's scale.
    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)

    log_score = (
        1.00 * np.log(endpoint_potential)
        + 1.40 * np.log(rank_affinity)
        + 0.45 * np.log(np.maximum(scale_compatibility, eps))
        - 0.50 * relative_length
        - 0.15 * global_length
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
