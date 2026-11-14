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

    positive = d[valid & (d > 0)]
    if positive.size:
        global_scale = max(
            float(np.median(positive)),
            np.finfo(float).eps,
        )
    else:
        global_scale = 1.0

    nearest_distance = np.min(safe_d, axis=1)
    local_scale = np.where(
        np.isfinite(nearest_distance) & (nearest_distance > 0),
        nearest_distance,
        global_scale,
    )

    k = min(4, n - 1)
    top_indices = np.argpartition(
        safe_d,
        kth=k - 1,
        axis=1,
    )[:, :k]

    rows = np.arange(n)[:, None]
    top_distances = safe_d[rows, top_indices]
    top_valid = valid[rows, top_indices]

    neighbor_scales = local_scale[top_indices]
    geometric_scale = np.sqrt(
        np.maximum(local_scale[:, None] * neighbor_scales, np.finfo(float).eps)
    )
    blended_scale = 0.5 * (geometric_scale + global_scale)

    local_ratio = np.minimum(top_distances / blended_scale, 40.0)
    global_ratio = np.minimum(top_distances / global_scale, 40.0)

    nearest_indices = np.argmin(safe_d, axis=1)
    mutual_nearest = (
        (top_indices == nearest_indices[:, None])
        & (nearest_indices[top_indices] == rows)
    )

    rank_weights = np.array([1.0, 0.74, 0.52, 0.36])[:k]
    candidate_score = (
        0.40 * np.exp(-global_ratio)
        + 0.32 * np.exp(-local_ratio)
        + 0.16 * rank_weights[None, :]
        + 0.12 * mutual_nearest
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
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
