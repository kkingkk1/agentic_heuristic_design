import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric stochastic edge-priority matrix for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must contain nonnegative values")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.ones((n, n), dtype=bool)
    np.fill_diagonal(valid, False)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps

    positive_edges = work[valid & (work > 0)]
    global_scale = (
        float(np.median(positive_edges))
        if positive_edges.size
        else 1.0
    )
    global_scale = max(global_scale, eps)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(work, order, axis=1)

    k = min(4, n - 1)
    neighborhood = sorted_distances[:, :k]
    finite_neighborhood = np.isfinite(neighborhood)

    local_values = np.where(
        finite_neighborhood,
        neighborhood,
        np.nan,
    )
    local_scale = np.nanmedian(local_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    global_affinity = np.exp(
        -np.clip(work / (1.15 * global_scale), 0.0, 50.0)
    )
    local_affinity = np.exp(
        -np.clip(work / pair_scale, 0.0, 50.0)
    )

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[np.arange(n), np.arange(n)] = np.inf

    finite_ranks = np.where(np.isfinite(ranks), ranks, n)

    endpoint_rank = 1.0 / (1.0 + 0.75 * finite_ranks)
    rank_affinity = np.sqrt(
        endpoint_rank * endpoint_rank.T
    )

    nearest = order[:, 0]
    nearest_mask = np.zeros((n, n), dtype=bool)
    nearest_mask[np.arange(n), nearest] = True
    nearest_mask[np.arange(n), np.arange(n)] = False

    top_three = (finite_ranks <= 2).astype(float)
    top_three[np.arange(n), np.arange(n)] = 0.0

    mutual_nearest = (
        nearest_mask & nearest_mask.T
    ).astype(float)
    mutual_top_three = top_three * top_three.T

    endpoint_sum = local_scale[:, None] + local_scale[None, :]
    long_edge = (
        (work > 2.25 * local_scale[:, None])
        & (work > 2.25 * local_scale[None, :])
    )
    long_penalty = np.where(long_edge, 0.55, 1.0)

    savings = np.maximum(endpoint_sum - work, 0.0)
    savings_scale = np.median(
        savings[valid & (savings > 0)]
    ) if np.any(valid & (savings > 0)) else global_scale
    savings_scale = max(float(savings_scale), eps)
    savings_signal = np.tanh(savings / savings_scale)

    score = (
        0.30 * local_affinity
        + 0.22 * global_affinity
        + 0.22 * rank_affinity
        + 0.14 * mutual_nearest
        + 0.08 * mutual_top_three
        + 0.04 * savings_signal
    )

    score *= long_penalty
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
