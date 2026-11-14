import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    finite_values = d[valid]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    scale = float(np.median(finite_values))
    scale = max(scale, np.finfo(float).eps)

    safe_d = np.where(valid, d, scale * 1e6)

    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    usable_count = np.maximum(valid.sum(axis=1), 1)
    local_closeness = np.clip(
        1.0 - ranks / usable_count[:, None],
        0.0,
        1.0,
    )
    local_closeness *= valid

    nearest = np.min(np.where(valid, d, np.inf), axis=1)
    sorted_distances = np.sort(np.where(valid, d, np.inf), axis=1)
    second_nearest = sorted_distances[:, 1]
    second_nearest = np.where(np.isfinite(second_nearest), second_nearest, nearest)
    regret = np.clip((second_nearest - nearest) / (second_nearest + scale), 0.0, 1.0)

    distance_score = np.exp(-safe_d / scale)
    mutual_support = np.sqrt(
        local_closeness * local_closeness.T
    )
    scarcity_bonus = 1.0 + 0.5 * (regret[:, None] + regret[None, :])

    score = distance_score * (1.0 + mutual_support) * scarcity_bonus
    score = 0.5 * (score + score.T)
    score *= valid & valid.T
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
