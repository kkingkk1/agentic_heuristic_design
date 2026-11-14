import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric stochastic-sampling prior for TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    safe = np.where(valid, distances, np.inf)
    eps = np.finfo(float).eps

    positive = safe[valid & (safe > 0.0)]
    global_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(safe, order, axis=1)

    nearest = sorted_distances[:, 0]
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    neighbor_count = min(3, n - 1)
    neighborhood = sorted_distances[:, :neighbor_count]
    neighborhood = np.where(
        np.isfinite(neighborhood),
        neighborhood,
        global_scale,
    )
    local_scale = np.mean(neighborhood, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    normalized_distance = np.clip(
        np.where(valid, distances, 0.0) / global_scale,
        0.0,
        700.0,
    )
    global_affinity = np.exp(-normalized_distance)
    global_affinity[~valid] = 0.0

    local_affinity = np.exp(
        -np.clip(
            np.where(valid, distances, 0.0) / pair_scale,
            0.0,
            700.0,
        )
    )
    local_affinity[~valid] = 0.0
    mutual_local = np.sqrt(local_affinity * local_affinity.T)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[~valid] = np.inf

    rank_affinity = 1.0 / np.power(
        1.0 + np.maximum(ranks, 0.0),
        0.8,
    )
    rank_affinity[~np.isfinite(rank_affinity)] = 0.0
    mutual_rank = np.sqrt(rank_affinity * rank_affinity.T)

    nearest_order = order[:, 0]
    mutual_nearest = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    for i, j in enumerate(nearest_order):
        if np.isfinite(safe[i, j]) and nearest_order[j] == i:
            mutual_nearest[i, j] = 1.0
    mutual_nearest = np.maximum(mutual_nearest, mutual_nearest.T)

    nearest_gap = np.maximum(
        sorted_distances[:, min(1, n - 1)] - nearest,
        0.0,
    )
    gap_scale = np.maximum(
        0.25 * local_scale,
        nearest_gap,
    )

    advantage_i = np.exp(
        -np.clip(
            np.maximum(
                np.where(valid, distances, 0.0)
                - nearest[:, None],
                0.0,
            )
            / (gap_scale[:, None] + eps),
            0.0,
            50.0,
        )
    )
    advantage_j = np.exp(
        -np.clip(
            np.maximum(
                np.where(valid, distances, 0.0)
                - nearest[None, :],
                0.0,
            )
            / (gap_scale[None, :] + eps),
            0.0,
            50.0,
        )
    )
    mutual_advantage = np.sqrt(advantage_i * advantage_j)
    mutual_advantage[~valid] = 0.0

    candidate_limit = min(4, n - 1)
    candidate_mask = (
        (ranks <= candidate_limit)
        | (ranks.T <= candidate_limit)
    ).astype(float)

    score = (
        0.40 * mutual_local
        + 0.25 * mutual_rank
        + 0.16 * mutual_advantage
        + 0.11 * mutual_nearest
        + 0.08 * np.sqrt(
            np.maximum(global_affinity, 0.0)
            * np.maximum(global_affinity.T, 0.0)
        )
    )

    score *= 0.86 + 0.14 * candidate_mask
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
