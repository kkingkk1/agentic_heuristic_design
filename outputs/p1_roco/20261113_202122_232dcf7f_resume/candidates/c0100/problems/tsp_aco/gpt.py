import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_input = np.isfinite(distances)
    distances = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    valid = finite_input & np.isfinite(distances)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    valid_values = distances[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif valid_values.size:
        scale = float(np.median(valid_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)
    safe = np.where(valid, distances, np.inf)
    np.fill_diagonal(safe, np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + eps)

    inverse_max = float(np.max(inverse))
    if not np.isfinite(inverse_max) or inverse_max <= 0.0:
        return np.zeros((n, n), dtype=float)

    inverse_scaled = inverse / inverse_max

    row_max = np.max(inverse, axis=1, keepdims=True)
    col_max = np.max(inverse, axis=0, keepdims=True)

    row_competitiveness = inverse / np.maximum(row_max, eps)
    col_competitiveness = inverse / np.maximum(col_max, eps)
    mutual_competitiveness = np.sqrt(
        row_competitiveness * col_competitiveness
    )

    local_size = min(3, n - 1)
    local_scale = np.full(n, scale, dtype=float)

    for i in range(n):
        neighbors = safe[i, valid[i]]
        if neighbors.size:
            nearest = np.partition(
                neighbors,
                min(local_size, neighbors.size) - 1,
            )[:local_size]
            estimate = float(np.mean(nearest))
            if np.isfinite(estimate) and estimate > 0.0:
                local_scale[i] = estimate

    local_scale = np.maximum(local_scale, eps)
    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    contrast = np.zeros((n, n), dtype=float)
    contrast[valid] = np.exp(
        -safe[valid] / (pair_scale[valid] + eps)
    )

    ranks = np.full((n, n), float(n), dtype=float)
    order = np.argsort(safe, axis=1, kind="stable")
    rank_values = np.arange(n, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = rank_values

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    attraction = np.exp(
        -np.minimum(safe, np.finfo(float).max) / scale
    )
    attraction[~valid] = 0.0
    attraction_max = float(np.max(attraction))
    if attraction_max > 0.0 and np.isfinite(attraction_max):
        attraction /= attraction_max

    score = (
        0.48 * attraction
        + 0.27 * mutual_competitiveness
        + 0.15 * mutual_rank
        + 0.10 * contrast
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    valid_count = float(np.sum(valid))
    exploration = 0.04
    score = (
        (1.0 - exploration) * score
        + exploration * valid.astype(float) / max(valid_count, 1.0)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
