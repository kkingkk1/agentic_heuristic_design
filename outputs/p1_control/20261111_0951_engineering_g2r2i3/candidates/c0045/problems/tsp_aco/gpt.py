import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return vectorized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, eps)

    positive_distances = np.where(valid & (d > 0.0), d, np.inf)
    k_scale = min(3, n - 1)
    nearest = np.partition(
        positive_distances,
        kth=k_scale - 1,
        axis=1,
    )[:, :k_scale]
    nearest = np.where(np.isfinite(nearest), nearest, fallback)

    local_scale = np.maximum(np.mean(nearest, axis=1), eps)
    normalized = np.divide(
        np.where(valid, d, np.inf),
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
    )

    clipped = np.minimum(normalized, 60.0)
    distance_affinity = np.exp(-clipped) / (1.0 + normalized)

    safe_distances = np.where(valid, d, np.inf)
    k_neighbor = min(4, n - 1)
    neighbor_indices = np.argpartition(
        safe_distances,
        kth=k_neighbor - 1,
        axis=1,
    )[:, :k_neighbor]

    nearest_mask = np.zeros((n, n), dtype=bool)
    row_indices = np.arange(n)[:, None]
    nearest_mask[row_indices, neighbor_indices] = True
    nearest_mask &= valid

    one_sided = nearest_mask | nearest_mask.T
    reciprocal = nearest_mask & nearest_mask.T
    neighborhood_bonus = (
        1.0
        + 0.30 * one_sided.astype(float)
        + 0.40 * reciprocal.astype(float)
    )

    mean_scale = max(float(np.mean(local_scale)), eps)
    relative_scale = np.clip(local_scale / mean_scale, 0.25, 4.0)
    bridge_factor = np.sqrt(
        relative_scale[:, None] * relative_scale[None, :]
    )
    bridge_bonus = 0.85 + 0.15 * np.clip(bridge_factor, 0.25, 2.5)

    score = distance_affinity * neighborhood_bonus * bridge_bonus
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    return np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    ).astype(float, copy=False)
