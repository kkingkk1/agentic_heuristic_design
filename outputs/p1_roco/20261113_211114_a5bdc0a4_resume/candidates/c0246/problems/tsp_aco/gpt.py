import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & ~diagonal
    valid = finite & finite.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(np.isfinite(raw), raw, 0.0)
    cleaned = np.maximum(cleaned, 0.0)
    distances = 0.5 * (cleaned + cleaned.T)

    values = distances[valid]
    positive = values[values > 0.0]
    if positive.size:
        scale = float(np.median(positive))
    else:
        nonzero = values[values != 0.0]
        scale = float(np.median(np.abs(nonzero))) if nonzero.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_valid = values[np.isfinite(values)]
    upper = (
        float(np.percentile(finite_valid, 95.0))
        if finite_valid.size
        else scale
    )
    cap = max(upper + 4.0 * scale, scale)
    distances = np.minimum(distances, cap)
    distances[~valid] = cap
    np.fill_diagonal(distances, cap)

    temperature = max(0.72 * scale, np.finfo(float).eps)
    affinity = np.exp(-np.minimum(distances, cap) / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    local_rank = np.zeros((n, n), dtype=float)
    endpoint_bonus = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[
            np.argsort(distances[i, candidates], kind="stable")
        ]
        rank_values = np.arange(order.size, dtype=float)
        denominator = max(float(order.size - 1), 1.0)

        local_rank[i, order] = 1.0 - rank_values / denominator

        best_distance = distances[i, order[0]]
        if order.size > 1:
            second_distance = distances[i, order[1]]
        else:
            second_distance = best_distance

        gap_scale = max(
            0.25 * scale,
            second_distance - best_distance,
            np.finfo(float).eps,
        )

        for j in candidates:
            if j == order[0]:
                gap = second_distance - distances[i, j]
                endpoint_bonus[i, j] = 1.0 - np.exp(
                    -max(gap, 0.0) / gap_scale
                )

    local_rank = np.sqrt(local_rank * local_rank.T)
    endpoint_bonus = np.sqrt(endpoint_bonus * endpoint_bonus.T)

    affinity_max = float(np.max(affinity))
    if not np.isfinite(affinity_max) or affinity_max <= 0.0:
        affinity_max = 1.0

    base = affinity / affinity_max
    score = (
        0.80 * base
        + 0.12 * local_rank
        + 0.08 * endpoint_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(16):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        balanced = balanced / row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.85 * score + 0.15 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
