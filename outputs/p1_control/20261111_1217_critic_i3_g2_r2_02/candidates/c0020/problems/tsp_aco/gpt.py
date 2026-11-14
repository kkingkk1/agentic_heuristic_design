import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-inclusion priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_off_diagonal = np.isfinite(distances)
    np.fill_diagonal(finite_off_diagonal, False)

    if not np.any(finite_off_diagonal):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[finite_off_diagonal]
    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        fallback = 1.0

    d = distances.copy()
    d[~np.isfinite(d)] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_positive = d[np.isfinite(d) & (d > 0.0)]
    global_scale = (
        float(np.median(finite_positive))
        if finite_positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    finite_mask = np.isfinite(d)
    row_counts = finite_mask.sum(axis=1)
    row_sums = np.where(finite_mask, d, 0.0).sum(axis=1)
    row_means = np.divide(
        row_sums,
        np.maximum(row_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=row_counts > 0,
    )

    nearest_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest_indices = np.argpartition(
        d,
        kth=nearest_count - 1,
        axis=1,
    )[:, :nearest_count]

    nearest_distances = np.take_along_axis(d, nearest_indices, axis=1)
    local_scales = np.max(nearest_distances, axis=1)
    local_scales[~np.isfinite(local_scales)] = row_means[
        ~np.isfinite(local_scales)
    ]
    local_scales = np.maximum(
        local_scales,
        global_scale * 1e-6,
    )

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

    local_order = np.argsort(
        nearest_distances,
        axis=1,
        kind="stable",
    )
    sorted_nearest = np.take_along_axis(
        nearest_indices,
        local_order,
        axis=1,
    )

    nearest = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    nearest[rows, sorted_nearest] = 1.0
    np.fill_diagonal(nearest, 0.0)

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[rows, sorted_nearest] = 1.0 / (
        1.0 + np.arange(nearest_count, dtype=float)[None, :]
    )

    mutual = nearest * nearest.T
    one_sided = np.clip(nearest + nearest.T - mutual, 0.0, 1.0)
    rank_symmetric = 0.5 * (rank_score + rank_score.T)

    score = (
        0.72 * locality * bridge
        + 0.16 * rank_symmetric
        + 0.10 * mutual
        + 0.02 * one_sided
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
