import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = valid & (d > 0.0)
    positive_d = np.where(positive, d, np.inf)
    nearest = np.min(positive_d, axis=1)

    finite_nearest = nearest[np.isfinite(nearest)]
    if finite_nearest.size:
        global_scale = float(np.median(finite_nearest))
    else:
        positive_values = d[positive]
        global_scale = (
            float(np.median(positive_values))
            if positive_values.size
            else 1.0
        )

    global_scale = max(global_scale, np.finfo(float).eps)
    row_scale = np.where(
        np.isfinite(nearest),
        nearest,
        global_scale,
    )

    pair_scale = np.sqrt(row_scale[:, None] * row_scale[None, :])
    local_ratio = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    bounded_local = np.minimum(local_ratio, 1e12)
    bounded_global = np.minimum(d / global_scale, 1e12)

    local_score = 1.0 / (1.0 + bounded_local)
    global_score = 1.0 / (1.0 + bounded_global)

    score = 0.58 * local_score + 0.22 * global_score

    k = min(3, n - 1)
    safe_d = np.where(valid, d, np.inf)
    candidates = np.argpartition(
        safe_d,
        kth=k - 1,
        axis=1,
    )[:, :k]

    candidate_distances = np.take_along_axis(
        safe_d,
        candidates,
        axis=1,
    )
    candidate_valid = np.take_along_axis(
        valid,
        candidates,
        axis=1,
    )

    order = np.argsort(
        candidate_distances,
        axis=1,
        kind="stable",
    )
    candidates = np.take_along_axis(candidates, order, axis=1)
    candidate_valid = np.take_along_axis(
        candidate_valid,
        order,
        axis=1,
    )

    rows = np.arange(n)
    rank_weights = (0.22, 0.11, 0.05)

    for rank, weight in enumerate(rank_weights[:k]):
        cols = candidates[:, rank]
        score[rows, cols] += weight * candidate_valid[:, rank]

    nearest_id = candidates[:, 0]
    nearest_ok = candidate_valid[:, 0]

    mutual_nearest = (
        nearest_ok
        & valid[rows, nearest_id]
        & (nearest_id[nearest_id] == rows)
    )

    mutual_bonus = np.zeros_like(score)
    mutual_bonus[rows, nearest_id] = mutual_nearest.astype(float)
    mutual_bonus = np.maximum(mutual_bonus, mutual_bonus.T)

    score += 0.12 * mutual_bonus
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
