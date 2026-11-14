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

    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    global_scale = max(float(np.median(positive)), eps)
    replacement = max(10.0 * float(np.max(positive)), global_scale)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_index = min(1, n - 2)
    local_scale = np.partition(
        work,
        neighbor_index,
        axis=1,
    )[:, neighbor_index]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = work / endpoint_scale
    global_length = work / global_scale

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_scale = float(max(n - 1, 1))
    normalized_rank = ranks / rank_scale
    bilateral_rank = np.exp(
        -1.15 * (normalized_rank + normalized_rank.T)
    )

    scale_log_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    scale_compatibility = np.exp(
        -0.35 * np.minimum(scale_log_gap, 50.0)
    )

    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )
    excess_longness = np.maximum(endpoint_longness - 1.0, 0.0)

    local_radius = 1.5 * local_scale[:, None]
    local_density = np.mean(valid & (work <= local_radius), axis=1)
    excess_density = np.maximum(local_density - 0.35, 0.0)
    density_penalty = (
        excess_density[:, None] + excess_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    reciprocal_length = global_scale / np.maximum(work, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_score = (
            0.72 * np.log(np.maximum(reciprocal_length, eps))
            + 1.05 * np.log(np.maximum(bilateral_rank, eps))
            + 0.35 * np.log(np.maximum(scale_compatibility, eps))
            - 0.72 * relative_length
            - 0.16 * global_length
            - 0.24 * excess_longness
            - 0.10 * density_penalty
        )

    finite_valid = valid & np.isfinite(log_score)
    if not np.any(finite_valid):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(log_score[finite_valid]))
    score = np.zeros((n, n), dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        score[finite_valid] = np.exp(
            np.clip(log_score[finite_valid] - shift, -745.0, 0.0)
        )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
