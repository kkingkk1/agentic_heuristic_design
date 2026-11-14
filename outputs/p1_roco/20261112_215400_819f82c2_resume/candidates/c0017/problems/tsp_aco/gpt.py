import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-prior indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square two-dimensional array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe_d = np.maximum(d, np.finfo(float).eps)

    inv = 1.0 / safe_d
    inv[~np.isfinite(inv)] = 0.0

    k = min(max(2, int(np.sqrt(n))), n - 1)
    nearest = np.partition(safe_d, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative = np.exp(
        -safe_d / np.sqrt(np.outer(local_scale, local_scale))
    )
    relative[~np.isfinite(relative)] = 0.0

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1).astype(float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~np.isfinite(rank_score)] = 0.0

    neighborhood_overlap = np.zeros((n, n), dtype=float)
    candidate_lists = np.argsort(safe_d, axis=1)[:, :k]
    for i in range(n):
        neighborhood_overlap[i] = np.isin(
            np.arange(n), candidate_lists[i]
        ).astype(float)
    bridge_score = 1.0 - (
        neighborhood_overlap @ neighborhood_overlap.T
    ) / max(k, 1)
    bridge_score = np.clip(bridge_score, 0.0, 1.0)

    adaptive = inv / (inv.sum(axis=1, keepdims=True) + np.finfo(float).eps)
    degree_balance = np.sqrt(adaptive * adaptive.T)

    heuristic = (
        0.35 * inv / scale
        + 0.30 * relative
        + 0.20 * (rank_score + rank_score.T) * 0.5
        + 0.15 * bridge_score
    )
    heuristic *= 0.5 + 0.5 * degree_balance
    heuristic = 0.5 * (heuristic + heuristic.T)

    heuristic[~np.isfinite(heuristic)] = 0.0
    np.fill_diagonal(heuristic, 0.0)

    maximum = float(np.max(heuristic))
    if maximum > 0.0:
        heuristic /= maximum

    return heuristic
