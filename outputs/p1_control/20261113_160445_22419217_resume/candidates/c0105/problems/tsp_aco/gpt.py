import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric consensus prior for stochastic TSP edge sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    safe = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    global_scale = max(global_scale, eps)

    # Estimate local density from several nearest valid neighbors.
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    sorted_distances = np.sort(safe, axis=1)
    neighborhood = sorted_distances[:, :k]
    neighborhood = np.where(
        np.isfinite(neighborhood),
        neighborhood,
        global_scale,
    )

    positive_neighborhood = np.where(
        neighborhood > 0.0,
        neighborhood,
        global_scale,
    )
    local_scale = np.median(positive_neighborhood, axis=1)
    local_scale = np.maximum(local_scale, eps)

    # Stable local ranks from each endpoint's perspective.
    order = np.argsort(safe, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[~valid] = np.inf

    finite_rank = np.where(valid, ranks + 1.0, np.inf)

    # Directional Gibbs support combines local and global distance scales.
    local_temperature = np.maximum(local_scale[:, None], eps)
    local_affinity = np.exp(
        -np.minimum(
            np.where(valid, d, 0.0) / local_temperature,
            50.0,
        )
    )
    local_affinity[~valid] = 0.0

    global_affinity = np.exp(
        -np.minimum(
            np.where(valid, d, 0.0) / global_scale,
            50.0,
        )
    )
    global_affinity[~valid] = 0.0

    directed_support = (
        0.70 * local_affinity
        + 0.30 * global_affinity
    )

    # Limited Sinkhorn-style competition correction.
    balanced = directed_support.copy()
    for _ in range(12):
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

    # Two-sided rank consensus favors edges accepted by both endpoints.
    rank_support = 1.0 / np.sqrt(
        np.maximum(finite_rank, 1.0)
        * np.maximum(finite_rank.T, 1.0)
    )
    rank_support[~valid] = 0.0

    # Opportunity-loss signal: an edge is valuable when alternatives are
    # substantially worse from one or both endpoints.
    nearest = sorted_distances[:, 0]
    second = (
        sorted_distances[:, 1]
        if n > 2
        else sorted_distances[:, 0]
    )

    nearest = np.where(np.isfinite(nearest), nearest, global_scale)
    second = np.where(np.isfinite(second), second, nearest)

    gaps = np.maximum(second - nearest, 0.0)
    positive_gaps = gaps[gaps > 0.0]
    gap_scale = (
        float(np.median(positive_gaps))
        if positive_gaps.size
        else global_scale
    )
    gap_scale = max(gap_scale, 0.1 * global_scale, eps)

    regret_i = np.exp(
        -np.minimum(
            np.maximum(np.where(valid, d, 0.0) - nearest[:, None], 0.0)
            / (gaps[:, None] + gap_scale),
            50.0,
        )
    )
    regret_j = np.exp(
        -np.minimum(
            np.maximum(np.where(valid, d, 0.0) - nearest[None, :], 0.0)
            / (gaps[None, :] + gap_scale),
            50.0,
        )
    )
    regret = np.sqrt(regret_i * regret_j)
    regret[~valid] = 0.0

    # Mutual-nearest edges receive a restrained structural bonus.
    nearest_index = np.argmin(safe, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest_index[i]
        if np.isfinite(safe[i, j]) and nearest_index[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    consensus = np.sqrt(balanced * balanced.T)

    score = (
        consensus
        * (0.55 + 0.45 * rank_support)
        * (0.70 + 0.30 * regret)
        * (1.0 + 0.12 * mutual_nearest)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
