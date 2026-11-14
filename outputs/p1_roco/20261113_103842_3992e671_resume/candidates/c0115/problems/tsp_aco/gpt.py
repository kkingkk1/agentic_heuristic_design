import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(d)
    positive = d > 0
    directed_valid = finite & positive
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid | directed_valid.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    effective_d = np.full((n, n), np.inf, dtype=float)

    both = directed_valid & directed_valid.T
    effective_d[both] = np.minimum(d[both], d.T[both])

    only_forward = directed_valid & ~directed_valid.T
    effective_d[only_forward] = d[only_forward]

    only_reverse = directed_valid.T & ~directed_valid
    effective_d[only_reverse] = d.T[only_reverse]

    valid_values = effective_d[valid]
    median = np.median(valid_values)
    mad = np.median(np.abs(valid_values - median))

    if not np.isfinite(median) or median <= 0:
        median = 1.0

    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
        robust_scale = median

    normalized = np.maximum(effective_d / median, 1e-12)
    log_distance = np.log1p(normalized)

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid] = 1.0 / (log_distance[valid] + 1e-12)

    rank_score = np.zeros((n, n), dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[np.argsort(effective_d[i, candidates], kind="stable")]
        nearest[i] = order[0]

        ranks = np.arange(1, order.size + 1, dtype=float)
        rank_score[i, order] = 1.0 / np.sqrt(ranks)

    row_max = np.max(rank_score, axis=1, keepdims=True)
    rank_score = rank_score / (row_max + 1e-12)

    reciprocal_rank = np.sqrt(rank_score * rank_score.T)

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    local_contrast = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        row_distances = effective_d[i, candidates]
        local_median = np.median(row_distances)
        contrast = np.maximum(local_median - row_distances, 0.0)
        local_contrast[i, candidates] = contrast / (
            robust_scale + local_median + 1e-12
        )

    local_contrast = 0.5 * (local_contrast + local_contrast.T)

    def normalize(matrix: np.ndarray) -> np.ndarray:
        values = matrix[valid]
        scale = np.percentile(values, 95) if values.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.max(values) if values.size else 1.0
        result = np.zeros_like(matrix, dtype=float)
        result[valid] = matrix[valid] / (scale + 1e-12)
        return result

    inverse_distance = normalize(inverse_distance)
    local_contrast = normalize(local_contrast)

    score = (
        0.52 * inverse_distance
        + 0.25 * reciprocal_rank
        + 0.15 * local_contrast
        + 0.08 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score[valid] += 1e-12

    return score
