import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-priority indicators for TSP."""

    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    eps = np.finfo(float).eps
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 4.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(work, order, axis=1)

    valid_counts = np.sum(valid, axis=1)
    best = sorted_distances[:, 0]

    second = np.where(
        valid_counts >= 2,
        sorted_distances[:, 1],
        best,
    )

    best = np.where(np.isfinite(best), best, global_scale)
    second = np.where(np.isfinite(second), second, best)

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        work,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    distance_term = np.exp(
        -np.clip(1.05 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.85 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.5
            * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.26 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )

    mutual_bonus = np.where(mutual_nearest, 1.14, 1.0)

    long_edge_penalty = np.exp(
        -0.60 * np.maximum(relative_distance - 1.0, 0.0)
    )

    neighborhood_size = min(3, max(1, n - 1))
    neighborhoods = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = order[i, :neighborhood_size]
        candidates = candidates[valid[i, candidates]]
        neighborhoods[i, candidates] = 1.0

    overlap = neighborhoods @ neighborhoods.T
    neighborhood_counts = np.maximum(
        neighborhoods.sum(axis=1),
        1.0,
    )

    normalized_overlap = np.divide(
        overlap,
        np.minimum(
            neighborhood_counts[:, None],
            neighborhood_counts[None, :],
        ),
        out=np.zeros((n, n), dtype=float),
        where=np.minimum(
            neighborhood_counts[:, None],
            neighborhood_counts[None, :],
        ) > 0.0,
    )

    bridge_strength = np.clip(1.0 - normalized_overlap, 0.0, 1.0)

    competitive_bridge = np.exp(
        -0.75 * np.maximum(relative_distance - 1.0, 0.0)
    )

    diversity_bonus = 1.0 + (
        0.10
        * bridge_strength
        * competitive_bridge
    )

    alternative = np.where(
        (nearest[:, None] == indices[None, :]),
        second[:, None],
        best[:, None],
    )

    reserve_quality = np.exp(
        -np.clip(
            0.5
            * (
                alternative / local_scale[:, None]
                + alternative.T / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    reserve_available = (
        (valid_counts[:, None] >= 2)
        & (valid_counts[None, :] >= 2)
    )

    reserve_bonus = 1.0 + np.where(
        reserve_available,
        0.08 * reserve_quality,
        0.0,
    )

    cluster_trap = (
        mutual_nearest
        & (normalized_overlap > 0.66)
    )

    trap_penalty = np.where(
        cluster_trap,
        0.94 + 0.06 * (1.0 - normalized_overlap),
        1.0,
    )

    scores = (
        distance_term
        * (
            0.52
            + 0.28 * bidirectional_rank
            + 0.20 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
        * diversity_bonus
        * reserve_bonus
        * trap_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
