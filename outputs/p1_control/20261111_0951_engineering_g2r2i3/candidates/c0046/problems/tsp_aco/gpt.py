import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    pair_valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(pair_valid, False)

    symmetric_distance = 0.5 * (d + d.T)

    positive = symmetric_distance[pair_valid & (symmetric_distance > 0.0)]
    fallback = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    fallback = max(fallback, np.finfo(float).eps)

    positive_rows = np.where(
        pair_valid & (symmetric_distance > 0.0),
        symmetric_distance,
        np.inf,
    )

    max_local_neighbors = min(4, n - 1)
    nearest = np.sort(positive_rows, axis=1)[:, :max_local_neighbors]
    nearest = np.where(np.isfinite(nearest), nearest, fallback)

    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    normalized_distance = np.divide(
        symmetric_distance,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
    )
    normalized_distance = np.where(
        pair_valid,
        normalized_distance,
        np.inf,
    )

    order = np.argsort(normalized_distance, axis=1)
    ranks = np.empty((n, n), dtype=float)
    rank_values = np.broadcast_to(
        np.arange(1, n + 1, dtype=float),
        (n, n),
    )
    np.put_along_axis(ranks, order, rank_values, axis=1)

    rank_affinity = 1.0 / np.sqrt(
        np.maximum(ranks * ranks.T, 1.0)
    )
    rank_affinity[~pair_valid] = 0.0

    median_scale = float(np.median(local_scale))
    median_scale = max(median_scale, np.finfo(float).eps)
    sparsity = np.clip(local_scale / median_scale, 0.5, 4.0)

    max_top_k = min(6, n - 1)
    adaptive_top_k = np.rint(
        3.0 + np.log2(sparsity)
    ).astype(int)
    adaptive_top_k = np.clip(adaptive_top_k, 2, max_top_k)

    rank_valid = ranks <= adaptive_top_k[:, None]
    mutual_top_k = rank_valid & rank_valid.T & pair_valid
    one_sided_top_k = (rank_valid | rank_valid.T) & pair_valid

    neighborhood_bonus = (
        1.0
        + 0.16 * one_sided_top_k.astype(float)
        + 0.34 * mutual_top_k.astype(float)
    )

    relative_sparsity = np.clip(
        local_scale / median_scale,
        0.35,
        3.0,
    )
    endpoint_sparsity = np.sqrt(
        relative_sparsity[:, None] * relative_sparsity[None, :]
    )
    bridge_bonus = 1.0 + 0.10 * np.clip(
        endpoint_sparsity - 1.0,
        -0.35,
        1.5,
    )

    distance_affinity = np.exp(
        -np.minimum(normalized_distance, 60.0)
    )
    distance_affinity /= 1.0 + normalized_distance

    score = (
        distance_affinity
        * (0.58 + 0.42 * rank_affinity)
        * neighborhood_bonus
        * bridge_bonus
    )

    score[~pair_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return score.astype(float, copy=False)
