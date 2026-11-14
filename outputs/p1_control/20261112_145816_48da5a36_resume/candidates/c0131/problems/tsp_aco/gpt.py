import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = 0.5 * (d + d.T)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d > 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    epsilon = np.finfo(float).eps
    fallback = max(float(np.median(positive_values)), epsilon)

    local_scale = np.full(n, fallback, dtype=float)
    for i in range(n):
        row_values = d[i, valid[i]]
        if row_values.size > 0:
            ordered = np.sort(row_values)
            local_scale[i] = ordered[min(1, ordered.size - 1)]

    local_scale = np.maximum(local_scale, epsilon)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = d / np.maximum(geometric_scale, epsilon)

    safe_d = np.where(valid, d, np.inf)
    order = np.argsort(safe_d, axis=1)
    sorted_distances = np.take_along_axis(safe_d, order, axis=1)

    ranks = np.empty_like(sorted_distances, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    valid_count = np.maximum(valid.sum(axis=1), 1)
    normalized_ranks = ranks / valid_count[:, None]
    endpoint_rank = np.exp(
        -1.8 * (normalized_ranks + normalized_ranks.T)
    )

    short_edge_score = np.exp(-0.85 * normalized_distance)

    scale_ratio = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + epsilon
    )
    scale_compatibility = np.exp(-0.75 * scale_ratio)

    local_density = np.sum(
        np.exp(-safe_d / local_scale[:, None]) * valid,
        axis=1,
    )
    sqrt_density = np.sqrt(local_density)
    crowding_penalty = 1.0 / np.sqrt(
        1.0 + 0.35 * (
            sqrt_density[:, None] + sqrt_density[None, :]
        )
    )

    score = (
        short_edge_score
        * endpoint_rank
        * scale_compatibility
        * crowding_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
