import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite & off_diagonal
    valid &= valid.T

    if not np.any(valid):
        return scores

    symmetric_distances = 0.5 * (distances + distances.T)
    positive = symmetric_distances[valid & (symmetric_distances > 0.0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = float(np.max(symmetric_distances[valid]))

    global_scale = max(global_scale, np.finfo(float).eps)

    work = np.where(valid, symmetric_distances, np.inf)
    order = np.argsort(work, axis=1, kind="stable")

    rank = np.full((n, n), float(n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    sorted_distances = np.sort(work, axis=1)
    nearest_count = min(3, n - 1)
    local_neighbors = sorted_distances[:, :nearest_count]

    local_neighbors = np.where(
        np.isfinite(local_neighbors),
        local_neighbors,
        global_scale,
    )
    local_scale = np.mean(local_neighbors, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    best = sorted_distances[:, 0]
    second = (
        sorted_distances[:, 1]
        if n > 2
        else sorted_distances[:, 0]
    )
    best = np.where(np.isfinite(best), best, global_scale)
    second = np.where(np.isfinite(second), second, best)

    scarcity = (second - best) / np.maximum(second, global_scale * 1e-12)
    scarcity = np.clip(scarcity, 0.0, 1.0)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :],
                   np.finfo(float).eps)
    )

    pair_distance = work / pair_scale
    distance_affinity = np.exp(
        -np.clip(0.82 * pair_distance, 0.0, 60.0)
    )

    rank_sum = rank + rank.T
    rank_affinity = np.exp(
        -np.clip(0.34 * rank_sum, 0.0, 60.0)
    )

    relative_to_endpoint = (
        work - np.minimum(best[:, None], best[None, :])
    ) / pair_scale
    relative_affinity = np.exp(
        -np.clip(0.55 * np.maximum(relative_to_endpoint, 0.0), 0.0, 60.0)
    )

    global_affinity = np.exp(
        -np.clip(0.22 * work / global_scale, 0.0, 60.0)
    )

    scarcity_bonus = 1.0 + 0.32 * (
        scarcity[:, None] + scarcity[None, :]
    )

    nearest = order[:, 0]
    mutual_nearest = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.10, 1.0)

    outlier_ratio = work / pair_scale
    outlier_penalty = 1.0 / (
        1.0 + 0.65 * np.maximum(outlier_ratio - 2.2, 0.0)
    )

    scores = (
        distance_affinity
        * (0.42 * rank_affinity
           + 0.34 * relative_affinity
           + 0.16 * global_affinity
           + 0.08 * distance_affinity)
        * scarcity_bonus
        * mutual_bonus
        * outlier_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
