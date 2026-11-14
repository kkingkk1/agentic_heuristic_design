import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & np.isfinite(raw.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.where(np.isfinite(raw), raw, 0.0)
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    values = distances[valid]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_values = values[np.isfinite(values)]
    upper = (
        float(np.percentile(finite_values, 95.0))
        if finite_values.size
        else scale
    )
    cap = max(upper + 3.0 * scale, scale)
    distances = np.minimum(distances, cap)
    distances[~valid] = cap
    np.fill_diagonal(distances, cap)

    local_scale = np.full(n, scale, dtype=float)
    local_best = np.full(n, cap, dtype=float)
    local_second = np.full(n, cap, dtype=float)
    ranks = np.zeros((n, n), dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[
            np.argsort(distances[i, candidates], kind="stable")
        ]
        nearest[i] = int(order[0])
        local_best[i] = distances[i, order[0]]

        if order.size > 1:
            local_second[i] = distances[i, order[1]]

        deviations = distances[i, order] - local_best[i]
        if deviations.size > 1:
            local_scale[i] = max(
                float(np.median(deviations[1:])),
                0.20 * scale,
                np.finfo(float).eps,
            )
        else:
            local_scale[i] = max(
                0.20 * scale,
                np.finfo(float).eps,
            )

        denominator = max(float(order.size - 1), 1.0)
        ranks[i, order] = 1.0 - np.arange(order.size) / denominator

    temperature = np.sqrt(local_scale[:, None] * local_scale[None, :])
    temperature = np.maximum(temperature, 0.35 * scale)

    affinity = np.exp(-distances / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if not np.isfinite(affinity_max) or affinity_max <= 0.0:
        affinity_max = 1.0
    affinity /= affinity_max

    rank_score = np.sqrt(ranks * ranks.T)

    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_bonus[i, j] = 1.0

    mutual_bonus = 0.5 * (mutual_bonus + mutual_bonus.T)

    gap = np.maximum(
        local_second[:, None] - distances,
        0.0,
    )
    gap_scale = np.maximum(local_scale[:, None], np.finfo(float).eps)
    endpoint_gap = 1.0 - np.exp(-gap / gap_scale)
    endpoint_gap[~valid] = 0.0
    endpoint_gap = np.sqrt(endpoint_gap * endpoint_gap.T)

    score = (
        0.78 * affinity
        + 0.12 * rank_score
        + 0.06 * endpoint_gap
        + 0.04 * mutual_bonus
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    row_sum = score.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, np.finfo(float).eps)
    row_normalized = score / row_sum
    row_normalized = 0.5 * (row_normalized + row_normalized.T)

    result = 0.90 * score + 0.10 * row_normalized
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
