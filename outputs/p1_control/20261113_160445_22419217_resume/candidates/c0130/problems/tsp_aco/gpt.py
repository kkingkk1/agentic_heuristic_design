import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    safe_d = np.where(valid, d, np.inf)
    nearest_distance = np.min(safe_d, axis=1)

    positive_values = d[valid & (d > 0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)
    local_scale = np.where(
        np.isfinite(nearest_distance) & (nearest_distance > 0),
        nearest_distance,
        global_scale,
    )

    k = min(4, n - 1)
    top_indices = np.argpartition(safe_d, kth=k - 1, axis=1)[:, :k]

    rows = np.arange(n)[:, None]
    top_distances = safe_d[rows, top_indices]
    top_valid = valid[rows, top_indices]

    local_ratio = top_distances / np.sqrt(
        local_scale[:, None] * local_scale[top_indices]
    )
    global_ratio = top_distances / global_scale

    local_score = np.exp(-np.minimum(local_ratio, 40.0))
    global_score = np.exp(-np.minimum(global_ratio, 40.0))

    rank_weights = np.array([1.0, 0.72, 0.50, 0.35])[:k]
    rank_score = rank_weights[None, :]

    nearest_indices = np.argmin(safe_d, axis=1)
    mutual_nearest = (
        top_indices == nearest_indices[:, None]
    ) & (
        nearest_indices[top_indices] == rows
    )

    candidate_score = (
        0.36 * global_score
        + 0.34 * local_score
        + 0.18 * rank_score
        + 0.12 * mutual_nearest.astype(float)
    )
    candidate_score = np.where(top_valid, candidate_score, 0.0)

    score = np.full((n, n), 0.015, dtype=float)
    score[~valid] = 0.0
    score[rows, top_indices] = np.maximum(
        score[rows, top_indices],
        candidate_score,
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
