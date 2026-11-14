import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric stochastic-sampling prior for TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, distances, np.inf)

    # Local scale from the second-nearest valid neighbor when available.
    sorted_distances = np.sort(safe, axis=1)
    kth = min(1, n - 2)
    local_scale = sorted_distances[:, kth]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    # Signal 1: globally and locally normalized short-edge preference.
    closeness = np.exp(-np.where(valid, distances, 0.0) / pair_scale)
    closeness = np.where(valid, closeness, 0.0)

    # Signal 2: reciprocal rank agreement from both endpoints.
    order = np.argsort(safe, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score = np.where(valid, rank_score, 0.0)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    # Signal 3: explicit mutual-nearest-neighbor reinforcement.
    nearest = np.argmin(safe, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if np.isfinite(safe[i, j]) and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Signal 4: savings relative to each endpoint's nearest alternative.
    nearest_distance = np.min(safe, axis=1)
    nearest_distance = np.where(
        np.isfinite(nearest_distance),
        nearest_distance,
        scale,
    )
    savings = np.maximum(
        nearest_distance[:, None] + nearest_distance[None, :]
        - np.where(valid, distances, np.inf),
        0.0,
    )
    positive_savings = savings[savings > 0.0]
    savings_scale = (
        float(np.median(positive_savings))
        if positive_savings.size
        else scale
    )
    savings_scale = max(savings_scale, np.finfo(float).eps)
    savings_score = np.tanh(savings / savings_scale)
    savings_score = np.where(valid, savings_score, 0.0)

    score = (
        0.45 * np.sqrt(closeness * closeness.T)
        + 0.25 * mutual_rank
        + 0.15 * mutual
        + 0.15 * savings_score
    )

    score = 0.5 * (score + score.T)
    score = np.where(valid, score, 0.0)
    np.fill_diagonal(score, 0.0)

    # Preserve sampling support for every node with at least one valid edge.
    for i in range(n):
        if not np.any(score[i] > 0.0):
            candidates = np.flatnonzero(valid[i])
            if candidates.size:
                best = candidates[np.argmin(distances[i, candidates])]
                score[i, best] = 1.0
                score[best, i] = 1.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
