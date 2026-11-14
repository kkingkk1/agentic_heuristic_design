import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-inclusion priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_mask = np.isfinite(distances)
    off_diagonal_mask = finite_mask.copy()
    np.fill_diagonal(off_diagonal_mask, False)

    if not np.any(off_diagonal_mask):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[off_diagonal_mask]
    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        fallback = 1.0

    d = distances.copy()
    d[~finite_mask] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_positive = d[np.isfinite(d) & (d > 0.0)]
    global_scale = (
        float(np.median(finite_positive))
        if finite_positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    finite_d = np.where(np.isfinite(d), d, 0.0)
    finite_counts = np.isfinite(d).sum(axis=1)
    row_sums = finite_d.sum(axis=1)
    row_means = np.divide(
        row_sums,
        np.maximum(finite_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=finite_counts > 0,
    )

    nearest_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    partitioned = np.partition(d, nearest_count - 1, axis=1)
    local_scales = partitioned[:, nearest_count - 1]
    local_scales[~np.isfinite(local_scales)] = row_means[
        ~np.isfinite(local_scales)
    ]
    local_scales = np.maximum(local_scales, global_scale * 1e-6)

    local_i = np.divide(
        d,
        local_scales[:, None],
        out=np.full_like(d, np.inf),
        where=np.isfinite(d),
    )
    local_j = np.divide(
        d,
        local_scales[None, :],
        out=np.full_like(d, np.inf),
        where=np.isfinite(d),
    )

    attraction_i = 1.0 / (1.0 + local_i)
    attraction_j = 1.0 / (1.0 + local_j)
    attraction_i[~np.isfinite(attraction_i)] = 0.0
    attraction_j[~np.isfinite(attraction_j)] = 0.0

    locality = np.sqrt(attraction_i * attraction_j)
    bridge = 0.90 + 0.20 * np.maximum(attraction_i, attraction_j)

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~np.isfinite(rank_score)] = 0.0
    rank_symmetric = 0.5 * (rank_score + rank_score.T)

    nearest_indices = np.argpartition(
        d,
        kth=nearest_count - 1,
        axis=1,
    )[:, :nearest_count]

    nearest = np.zeros((n, n), dtype=bool)
    nearest[rows, nearest_indices] = True
    np.fill_diagonal(nearest, False)

    mutual = nearest & nearest.T

    score = (
        0.72 * locality * bridge
        + 0.16 * rank_symmetric
        + 0.10 * mutual.astype(float)
    )

    directed_weight = 0.02
    flat_rows = np.broadcast_to(
        np.arange(n)[:, None],
        nearest_indices.shape,
    ).ravel()
    flat_cols = nearest_indices.ravel()

    score[flat_rows, flat_cols] += directed_weight
    score[flat_cols, flat_rows] += directed_weight

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
