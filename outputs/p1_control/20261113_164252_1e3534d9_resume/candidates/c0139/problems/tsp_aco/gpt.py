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
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    undirected_d = 0.5 * (d + d.T)
    safe_d = np.where(valid, undirected_d, np.inf)

    positive = safe_d[valid & (safe_d > 0)]
    eps = np.finfo(float).eps
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    nearest_distance = np.min(safe_d, axis=1)
    local_floor = max(0.25 * global_scale, eps)
    local_scale = np.where(
        np.isfinite(nearest_distance),
        np.maximum(nearest_distance, local_floor),
        global_scale,
    )

    k = min(4, n - 1)
    rows = np.arange(n)[:, None]
    top_indices = np.argpartition(safe_d, kth=k - 1, axis=1)[:, :k]
    top_distances = safe_d[rows, top_indices]
    top_valid = valid[rows, top_indices]

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[top_indices])
    local_ratio = np.clip(
        top_distances / np.maximum(pair_scale, eps),
        0.0,
        40.0,
    )
    global_ratio = np.clip(top_distances / global_scale, 0.0, 40.0)

    nearest_indices = np.argmin(safe_d, axis=1)
    mutual_nearest = (
        (top_indices == nearest_indices[:, None])
        & (nearest_indices[top_indices] == rows)
        & top_valid
    )

    rank_weights = np.array([1.0, 0.72, 0.50, 0.35], dtype=float)[:k]
    rank_score = np.broadcast_to(rank_weights, (n, k))

    candidate_score = (
        0.36 * np.exp(-global_ratio)
        + 0.34 * np.exp(-local_ratio)
        + 0.18 * rank_score
        + 0.12 * mutual_nearest.astype(float)
    )
    candidate_score = np.where(top_valid, candidate_score, 0.0)

    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_mask[rows, top_indices] = top_valid

    score = np.zeros((n, n), dtype=float)
    score[rows, top_indices] = candidate_score

    candidate_values = candidate_score[top_valid]
    reference = float(np.median(candidate_values)) if candidate_values.size else 1.0
    exploration_prior = 0.08 * max(reference, eps)

    score = np.where(
        valid & ~candidate_mask,
        exploration_prior,
        score,
    )
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
