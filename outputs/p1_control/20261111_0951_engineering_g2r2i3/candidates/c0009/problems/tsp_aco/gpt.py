import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps

    positive_distances = np.where(valid & (d > 0.0), d, np.inf)
    sorted_distances = np.sort(positive_distances, axis=1)

    k = min(3, n - 1)
    nearest = sorted_distances[:, :k]
    local_scale = np.where(np.isfinite(nearest), nearest, fallback)
    local_scale = np.mean(local_scale, axis=1)
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    safe_distance = np.where(valid, np.maximum(d, 0.0), np.inf)

    normalized_distance = np.divide(
        safe_distance,
        endpoint_scale,
        out=np.full_like(safe_distance, np.inf, dtype=float),
        where=np.isfinite(safe_distance),
    )

    local_affinity = np.exp(-normalized_distance)
    mutual_affinity = np.exp(-0.5 * normalized_distance)

    rank_order = np.argsort(np.argsort(safe_distance, axis=1), axis=1)
    rank_weight = 1.0 / (1.0 + rank_order.astype(float))
    rank_weight = rank_weight / np.maximum(
        np.max(rank_weight, axis=1, keepdims=True), eps
    )
    mutual_rank = np.sqrt(rank_weight * rank_weight.T)

    density = local_scale / max(float(np.median(local_scale)), eps)
    bridge_bonus = np.sqrt(density[:, None] * density[None, :])
    bridge_bonus = np.clip(bridge_bonus, 0.5, 2.0)

    score = (
        local_affinity
        * (0.55 + 0.45 * mutual_affinity)
        * (0.70 + 0.30 * mutual_rank)
        * (0.85 + 0.15 * bridge_bonus)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return score.astype(float, copy=False)
