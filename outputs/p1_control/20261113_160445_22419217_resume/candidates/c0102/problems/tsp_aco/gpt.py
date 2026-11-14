import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a stable symmetric edge-priority matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return result

    safe = np.where(valid, d, np.inf)
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    k = min(4, n - 1)
    sorted_distances = np.sort(safe, axis=1)
    neighborhood = sorted_distances[:, :k]
    neighborhood_valid = np.isfinite(neighborhood)

    local_sum = np.sum(
        np.where(neighborhood_valid, neighborhood, 0.0),
        axis=1,
    )
    local_count = np.sum(neighborhood_valid, axis=1)

    local_scale = np.divide(
        local_sum,
        local_count,
        out=np.full(n, global_scale, dtype=float),
        where=local_count > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    nearest = sorted_distances[:, 0]
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    if n > 2:
        second = sorted_distances[:, 1]
        second = np.where(np.isfinite(second), second, nearest)
    else:
        second = nearest

    gaps = np.maximum(second - nearest, 0.0)
    positive_gaps = gaps[gaps > 0.0]
    gap_scale = (
        float(np.median(positive_gaps))
        if positive_gaps.size
        else global_scale
    )
    gap_scale = max(gap_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks.fill(np.inf)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n, dtype=float)[None, :]
    ranks[~valid] = np.inf

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    global_affinity = np.exp(
        -np.clip(safe / global_scale, 0.0, 60.0)
    )
    local_affinity = np.exp(
        -np.clip(safe / pair_scale, 0.0, 60.0)
    )

    rank_support = 1.0 / np.sqrt(
        (1.0 + np.maximum(ranks, 0.0))
        * (1.0 + np.maximum(ranks.T, 0.0))
    )

    endpoint_gap_scale = np.maximum(
        np.maximum(gaps, 0.25 * local_scale),
        0.1 * gap_scale,
    )

    excess_i = np.maximum(
        safe - nearest[:, None],
        0.0,
    )
    excess_j = np.maximum(
        safe - nearest[None, :],
        0.0,
    )

    regret_support = np.exp(
        -0.5
        * (
            np.clip(
                excess_i / endpoint_gap_scale[:, None],
                0.0,
                60.0,
            )
            + np.clip(
                excess_j / endpoint_gap_scale[None, :],
                0.0,
                60.0,
            )
        )
    )

    mutual_nearest = (
        (ranks == 0.0)
        & (ranks.T == 0.0)
        & valid
    ).astype(float)

    local_threshold = 1.8 * pair_scale
    long_edge_suppression = np.where(
        safe > local_threshold,
        0.55,
        1.0,
    )

    base_affinity = np.sqrt(global_affinity * local_affinity)

    score = base_affinity * (
        0.58
        + 0.18 * rank_support
        + 0.16 * regret_support
        + 0.08 * mutual_nearest
    )

    score *= long_edge_suppression
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
