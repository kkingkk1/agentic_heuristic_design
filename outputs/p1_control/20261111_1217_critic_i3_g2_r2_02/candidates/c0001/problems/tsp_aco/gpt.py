import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d >= 0)
    valid &= ~np.eye(n, dtype=bool)

    safe_d = np.where(valid, d, np.nan)
    row_scale = np.nanmedian(safe_d, axis=1)
    global_scale = np.nanmedian(safe_d)

    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        global_scale,
    )

    scale = np.sqrt(row_scale[:, None] * row_scale[None, :])
    scale = np.maximum(scale, np.finfo(float).eps)

    relative_distance = np.where(valid, d / scale, np.inf)
    direct_score = np.exp(-relative_distance)
    direct_score[~valid] = 0.0

    rank_score = np.zeros((n, n), dtype=float)
    mutual_neighbor = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[np.argsort(d[i, candidates], kind="stable")]
        ranks = np.empty(order.size, dtype=float)
        ranks[:] = np.arange(order.size, dtype=float)
        rank_score[i, order] = np.exp(-ranks / max(1.0, 0.35 * order.size))

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            nearest[i] = candidates[np.argmin(d[i, candidates])]

    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_neighbor[i, j] = 1.0

    row_strength = np.sum(direct_score, axis=1, keepdims=True)
    row_strength = np.maximum(row_strength, np.finfo(float).eps)

    competition = direct_score / row_strength
    competition = np.sqrt(competition * competition.T)

    score = (
        0.50 * direct_score
        + 0.30 * rank_score
        + 0.15 * competition
        + 0.05 * mutual_neighbor
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
