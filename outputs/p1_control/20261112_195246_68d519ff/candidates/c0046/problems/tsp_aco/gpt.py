import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.array(distances, copy=True)

    finite = np.isfinite(d)
    valid_positive = finite & (d > 0)

    if np.any(valid_positive):
        fallback = float(np.max(d[valid_positive]))
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 1.0
    else:
        fallback = 1.0

    d[~finite] = fallback
    d[d < 0] = fallback

    # Use a symmetric representation because TSP edge priors are undirected.
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_scale = max(float(np.median(positive)), eps)

    # Local scale from the closest neighbors of each node.
    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    ordered = np.sort(d, axis=1)
    local_scale = np.mean(ordered[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / np.maximum(endpoint_scale, eps)

    local_affinity = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    # Rank-based evidence from both endpoints.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(float(n - 1), 1.0)
    rank_affinity = 1.0 - ranks / rank_scale
    rank_affinity = np.clip(rank_affinity, 0.0, 1.0)

    symmetric_rank = 0.5 * (rank_affinity + rank_affinity.T)
    mutual_rank = np.sqrt(
        np.maximum(rank_affinity, 0.0)
        * np.maximum(rank_affinity.T, 0.0)
    )

    # Global distance evidence prevents local scales from dominating entirely.
    global_affinity = np.exp(-np.clip(d / positive_scale, 0.0, 50.0))

    score = (
        0.45 * local_affinity
        + 0.20 * symmetric_rank
        + 0.20 * mutual_rank
        + 0.15 * global_affinity
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.clip(score, 0.0, 1.0).astype(float, copy=False)
