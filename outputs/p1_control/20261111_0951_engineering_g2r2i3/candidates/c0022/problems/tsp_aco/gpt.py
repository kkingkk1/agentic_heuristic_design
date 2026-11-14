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

    safe_d = np.where(valid, d, fallback)
    safe_d = np.where(safe_d > 0.0, safe_d, fallback)
    np.fill_diagonal(safe_d, fallback)

    k = min(3, n - 1)
    nearest = np.partition(safe_d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.maximum(np.mean(nearest, axis=1), eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = safe_d / np.maximum(pair_scale, eps)

    # Fast short-edge affinity without repeated exponential evaluations.
    score = 1.0 / (1.0 + normalized_distance)

    # Mildly favor edges connected to locally sparse nodes.
    density_ratio = local_scale / np.maximum(
        np.mean(local_scale), eps
    )
    bridge_bonus = np.sqrt(density_ratio[:, None] * density_ratio[None, :])
    bridge_bonus = np.clip(bridge_bonus, 0.5, 2.0)
    score *= 0.9 + 0.1 * bridge_bonus

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    return np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    ).astype(float, copy=False)
