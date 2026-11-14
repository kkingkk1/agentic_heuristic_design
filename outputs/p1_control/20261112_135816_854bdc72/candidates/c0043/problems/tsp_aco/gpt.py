import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-prior indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances)

    valid_positive = finite & off_diagonal & (distances > 0)
    positive_values = distances[valid_positive]

    if positive_values.size == 0:
        return np.zeros_like(distances, dtype=float)

    fallback = float(np.max(positive_values))
    fallback = max(fallback, np.finfo(float).eps)

    safe = np.where(
        finite & off_diagonal & (distances >= 0),
        distances,
        fallback,
    )
    safe = 0.5 * (safe + safe.T)
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe) & off_diagonal
    positive = safe[valid & (safe > 0)]
    global_scale = float(np.median(positive)) if positive.size else fallback
    global_scale = max(global_scale, np.finfo(float).eps)

    positive_safe = np.where(valid & (safe > 0), safe, np.nan)
    local_scale = np.nanmedian(positive_safe, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    epsilon = max(np.finfo(float).eps, global_scale * 1e-12)

    inverse_distance = 1.0 / (safe + epsilon)
    inverse_distance[~valid] = 0.0
    inverse_max = float(np.max(inverse_distance))
    if inverse_max > 0:
        inverse_distance /= inverse_max

    relative_distance = safe / np.sqrt(
        np.outer(local_scale, local_scale)
    )
    relative_closeness = np.exp(-np.minimum(relative_distance, 60.0))
    relative_closeness[~valid] = 0.0

    nearest_distance = np.min(
        np.where(valid, safe, np.inf),
        axis=1,
    )
    nearest_distance = np.where(
        np.isfinite(nearest_distance) & (nearest_distance > 0),
        nearest_distance,
        epsilon,
    )

    reciprocal_preference = np.exp(
        -np.minimum(
            safe / nearest_distance[:, None]
            + safe / nearest_distance[None, :],
            60.0,
        )
    )
    reciprocal_preference[~valid] = 0.0

    rank_order = np.argsort(
        np.where(valid, safe, np.inf),
        axis=1,
        kind="stable",
    )
    ranks = np.empty_like(rank_order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, rank_order[i]] = rank_values

    rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))
    rank_score[~valid] = 0.0
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    competition = np.sum(relative_closeness, axis=1)
    competition = np.maximum(competition, epsilon)
    isolation = 1.0 / np.sqrt(
        competition[:, None] * competition[None, :]
    )
    isolation /= max(float(np.max(isolation)), epsilon)

    score = (
        np.maximum(inverse_distance, epsilon) ** 0.30
        * np.maximum(relative_closeness, epsilon) ** 0.30
        * np.maximum(reciprocal_preference, epsilon) ** 0.20
        * np.maximum(mutual_rank, epsilon) ** 0.10
        * np.maximum(isolation, epsilon) ** 0.10
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score.astype(float, copy=False)
