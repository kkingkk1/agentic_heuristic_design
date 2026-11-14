import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive_mask = valid & (d > 0.0)
    positive_values = d[positive_mask]
    fallback = float(np.median(positive_values)) if positive_values.size else 1.0
    fallback = max(fallback, eps)

    k_scale = min(3, n - 1)
    positive_rows = np.where(positive_mask, d, np.inf)
    nearest_positive = np.partition(
        positive_rows,
        kth=k_scale - 1,
        axis=1,
    )[:, :k_scale]
    nearest_positive = np.where(
        np.isfinite(nearest_positive),
        nearest_positive,
        fallback,
    )

    local_scale = np.maximum(np.mean(nearest_positive, axis=1), eps)
    scale_matrix = np.sqrt(local_scale[:, None] * local_scale[None, :])

    normalized_distance = np.divide(
        np.where(valid, d, np.inf),
        scale_matrix,
    )

    clipped_distance = np.minimum(normalized_distance, 60.0)
    distance_affinity = np.exp(-clipped_distance)
    distance_affinity /= 1.0 + normalized_distance

    order = np.argsort(
        np.where(valid, normalized_distance, np.inf),
        axis=1,
        kind="stable",
    )

    ranks = np.empty((n, n), dtype=float)
    rank_values = np.arange(1.0, n + 1.0)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = rank_values[None, :]
    rank_affinity = 1.0 / np.sqrt(ranks)
    rank_affinity[~valid] = 0.0

    k_neighbor = min(4, n - 1)
    nearest_idx = np.argpartition(
        np.where(valid, normalized_distance, np.inf),
        kth=k_neighbor - 1,
        axis=1,
    )[:, :k_neighbor]

    nearest_mask = np.zeros((n, n), dtype=bool)
    nearest_mask[row_ids, nearest_idx] = valid[row_ids, nearest_idx]

    reciprocal = nearest_mask & nearest_mask.T
    one_sided = nearest_mask | nearest_mask.T
    neighborhood_bonus = (
        1.0
        + 0.35 * one_sided.astype(float)
        + 0.45 * reciprocal.astype(float)
    )

    mean_scale = max(float(np.mean(local_scale)), eps)
    relative_sparsity = np.clip(local_scale / mean_scale, 0.25, 4.0)
    pair_sparsity = np.sqrt(
        relative_sparsity[:, None] * relative_sparsity[None, :]
    )
    bridge_bonus = 0.8 + 0.2 * np.clip(pair_sparsity, 0.25, 2.5)

    score = (
        distance_affinity
        * (0.65 + 0.35 * rank_affinity)
        * neighborhood_bonus
        * bridge_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    return np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    ).astype(float, copy=False)
