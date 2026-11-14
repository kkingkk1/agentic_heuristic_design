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

    if not np.any(finite):
        return scores

    # Build a symmetric undirected distance matrix.
    forward = np.where(finite, distances, np.inf)
    reverse = forward.T
    both = np.isfinite(forward) & np.isfinite(reverse)
    either = np.isfinite(forward) | np.isfinite(reverse)

    d = np.full((n, n), np.inf, dtype=float)
    d[both] = 0.5 * (forward[both] + reverse[both])
    d[either & ~both] = np.minimum(forward[either & ~both], reverse[either & ~both])
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    if not np.any(valid):
        return scores

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.max(finite_values)), 1.0)

    global_scale = max(global_scale, np.finfo(float).eps)
    maximum_distance = max(float(np.max(finite_values)), global_scale)

    # Finite replacement keeps unavailable edges from creating NaNs.
    penalty = 4.0 * maximum_distance + global_scale
    work = np.where(valid, d, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(work, order, axis=1)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    # Use several nearest neighbors to obtain a robust local scale.
    neighbor_count = min(5, n - 1)
    nearest_block = sorted_distances[:, :neighbor_count]
    local_scale = np.median(nearest_block, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Isolated nodes receive a larger bonus for their relatively necessary edges.
    isolation = local_scale / max(float(np.median(local_scale)), global_scale)
    isolation = np.clip(isolation, 0.25, 4.0)

    endpoint_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = work / endpoint_scale

    distance_affinity = np.exp(-np.clip(normalized_distance, 0.0, 60.0))

    # Exponential rank affinity is smoother than a reciprocal rank.
    rank_affinity = np.exp(
        -0.55 * (ranks + ranks.T) / max(float(n - 1), 1.0)
    )

    row_quality = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 60.0)
    )
    col_quality = np.exp(
        -np.clip(work / local_scale[None, :], 0.0, 60.0)
    )
    endpoint_quality = np.sqrt(row_quality * col_quality)

    # Regret emphasizes edges that are important to locally constrained nodes.
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else sorted_distances[:, 0]
    regret = (second - best) / np.maximum(second, np.finfo(float).eps)
    regret = np.clip(regret, 0.0, 1.0)

    scarcity = np.sqrt(
        np.clip(isolation[:, None], 0.25, 4.0)
        * np.clip(isolation[None, :], 0.25, 4.0)
    )
    scarcity_bonus = 1.0 + 0.18 * np.tanh(scarcity - 1.0)
    regret_bonus = 1.0 + 0.30 * (regret[:, None] + regret[None, :])

    # Prefer mutual nearest-neighbor edges while retaining exploration.
    nearest = order[:, 0]
    mutual = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )
    mutual_bonus = np.where(mutual, 1.22, 1.0)

    # Suppress edges that are poor relative to both endpoint neighborhoods.
    outlier_excess = np.maximum(normalized_distance - 2.0, 0.0)
    outlier_penalty = 1.0 / (1.0 + 0.75 * outlier_excess)

    scores = (
        distance_affinity
        * (0.45 + 0.25 * rank_affinity + 0.30 * endpoint_quality)
        * scarcity_bonus
        * regret_bonus
        * mutual_bonus
        * outlier_penalty
    )

    # Additive exploration floor prevents complete loss of valid alternatives.
    scores = 0.94 * scores + 0.06 * endpoint_quality

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
