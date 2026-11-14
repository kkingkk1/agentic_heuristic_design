import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, regret-aware edge-priority matrix for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0.0):
        raise ValueError("distance_matrix must contain nonnegative values")

    eps = np.finfo(float).eps
    valid = ~np.eye(n, dtype=bool)

    d = distances.copy()
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[~valid] = np.inf

    sorted_d = np.sort(d, axis=1)
    nearest = sorted_d[:, 0]
    second = sorted_d[:, min(1, n - 2)]
    third = sorted_d[:, min(2, n - 2)]

    nearest = np.where(np.isfinite(nearest), nearest, global_scale)
    second = np.where(np.isfinite(second), second, nearest)
    third = np.where(np.isfinite(third), third, second)

    local_scale = 0.55 * second + 0.45 * third
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    distance_affinity = np.exp(
        -np.clip(d / (0.9 * pair_scale), 0.0, 700.0)
    )
    distance_affinity[~valid] = 0.0

    rank_affinity = np.exp(
        -np.clip(ranks / max(2.0, np.sqrt(float(n))), 0.0, 700.0)
    )
    rank_affinity[~valid] = 0.0

    mutual_distance = np.sqrt(
        distance_affinity * distance_affinity.T
    )
    mutual_rank = np.sqrt(rank_affinity * rank_affinity.T)

    row_gap = np.maximum(second - nearest, 0.0)
    gap_values = row_gap[row_gap > 0.0]
    gap_scale = float(np.median(gap_values)) if gap_values.size else global_scale
    gap_scale = max(gap_scale, eps)

    row_regret = np.tanh(row_gap / gap_scale)
    regret = np.sqrt(
        np.maximum(row_regret[:, None], 0.0)
        * np.maximum(row_regret[None, :], 0.0)
    )

    nearest_index = np.argmin(d, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    cols = nearest_index
    reciprocal = nearest_index[nearest_index] == rows
    mutual_nearest[rows[reciprocal], cols[reciprocal]] = 1.0
    mutual_nearest[cols[reciprocal], rows[reciprocal]] = 1.0
    np.fill_diagonal(mutual_nearest, 0.0)

    affinity = (
        0.48 * mutual_distance
        + 0.28 * mutual_rank
        + 0.12 * regret
        + 0.12 * mutual_nearest
    )
    affinity[~valid] = 0.0

    balanced = affinity.copy()
    for _ in range(24):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=col_sum > 0.0,
        )

    score = 0.7 * affinity + 0.3 * balanced
    score *= 1.0 + 0.25 * regret
    score *= 1.0 + 0.10 * mutual_nearest

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
