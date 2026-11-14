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
    np.fill_diagonal(finite, False)

    # Keep only edges available from both directions for an undirected instance.
    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = np.full((n, n), np.inf, dtype=float)
    d[valid] = 0.5 * (distances[valid] + distances.T[valid])
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive = finite_values[finite_values > 0.0]
    scale = (
        float(np.median(positive))
        if positive.size
        else max(float(np.max(finite_values)), 1.0)
    )
    scale = max(scale, np.finfo(float).eps)

    # Replace missing edges by a large but finite distance for ranking only.
    max_distance = max(float(np.max(finite_values)), scale)
    penalty = max_distance * 8.0 + scale
    work = np.where(valid, d, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(work, order, axis=1)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    available_count = np.maximum(valid.sum(axis=1), 1)
    k = np.minimum(np.maximum(3, available_count // 3), 8)

    local_scales = np.empty(n, dtype=float)
    for i in range(n):
        count = int(available_count[i])
        usable = sorted_distances[i, :count]
        kk = min(max(2, int(k[i])), count)
        local_scales[i] = float(np.median(usable[:kk]))

    local_scales = np.maximum(local_scales, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scales[:, None] * local_scales[None, :])
    relative_distance = work / pair_scale

    distance_term = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    # Smooth candidate preference. Close-ranked edges receive more weight.
    rank_decay = np.sqrt(
        (1.0 + ranks) / available_count[:, None]
        * (1.0 + ranks.T) / available_count[None, :]
    )
    rank_term = 1.0 / (1.0 + rank_decay)

    endpoint_quality = np.sqrt(
        np.exp(-np.clip(work / local_scales[:, None], 0.0, 50.0))
        * np.exp(-np.clip(work / local_scales[None, :], 0.0, 50.0))
    )

    # Preserve edges incident to nodes whose alternatives are relatively poor.
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 1 else best
    regret = (second - best) / np.maximum(second, np.finfo(float).eps)
    regret = np.clip(regret, 0.0, 1.0)
    regret_bonus = 1.0 + 0.45 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    mutual_nearest = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.30, 1.0)

    # Favor edges that are good for both endpoints, while retaining exploration.
    candidate_bonus = np.exp(
        -0.35 * np.maximum(
            ranks / np.maximum(available_count[:, None] - 1, 1),
            ranks.T / np.maximum(available_count[None, :] - 1, 1),
        )
    )

    outlier_penalty = 1.0 / (
        1.0 + np.maximum(relative_distance - 2.0, 0.0) ** 1.35
    )

    scores = (
        distance_term
        * (0.40 + 0.24 * rank_term + 0.22 * endpoint_quality
           + 0.14 * candidate_bonus)
        * regret_bonus
        * mutual_bonus
        * outlier_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
