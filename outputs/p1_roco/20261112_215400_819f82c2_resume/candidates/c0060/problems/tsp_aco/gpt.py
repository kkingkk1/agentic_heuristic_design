import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    raw = distance_matrix.copy()
    finite = np.isfinite(raw)
    np.fill_diagonal(finite, False)

    positive = raw[finite & (raw > 0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    raw[~finite] = np.inf
    raw[raw < 0] = 0.0
    np.fill_diagonal(raw, np.inf)

    finite_t = np.isfinite(raw.T)
    both_finite = finite & finite_t

    symmetric_distance = np.full((n, n), np.inf, dtype=float)
    symmetric_distance[both_finite] = (
        raw[both_finite] + raw.T[both_finite]
    ) * 0.5

    only_forward = finite & ~finite_t
    only_backward = finite_t & ~finite
    symmetric_distance[only_forward] = raw[only_forward]
    symmetric_distance[only_backward] = raw.T[only_backward]
    np.fill_diagonal(symmetric_distance, np.inf)

    valid = np.isfinite(symmetric_distance)
    if not np.any(valid):
        return scores

    row_values = np.where(valid, symmetric_distance, np.nan)

    row_median = np.nanmedian(row_values, axis=1)
    row_min = np.nanmin(row_values, axis=1)

    row_median[~np.isfinite(row_median)] = global_scale
    row_min[~np.isfinite(row_min)] = global_scale

    local_scale = 0.5 * row_median + 0.5 * row_min
    local_scale[~np.isfinite(local_scale)] = global_scale
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    normalized = symmetric_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    finite_normalized = np.where(valid, normalized, np.inf)
    preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row = finite_normalized[i]
        mask = np.isfinite(row)

        if not np.any(mask):
            continue

        logits = np.zeros(n, dtype=float)
        logits[mask] = -np.clip(row[mask], 0.0, 80.0)

        maximum = np.max(logits[mask])
        weights = np.zeros(n, dtype=float)
        weights[mask] = np.exp(logits[mask] - maximum)

        total = weights.sum()
        if total > 0:
            preferences[i] = weights / total

    bidirectional_preference = np.sqrt(
        preferences * preferences.T
    )

    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        mask = valid[i]
        count = int(np.sum(mask))

        if count == 0:
            continue

        order = np.argsort(symmetric_distance[i])
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n, dtype=float)

        ranks[i, mask] = 1.0 / (1.0 + rank[mask])

    rank_continuity = np.sqrt(ranks * ranks.T)

    nearest = np.zeros((n, n), dtype=float)
    second_nearest = np.zeros(n, dtype=float)

    for i in range(n):
        candidates = np.where(valid[i])[0]
        if candidates.size == 0:
            continue

        ordered = candidates[
            np.argsort(symmetric_distance[i, candidates])
        ]
        nearest[i, ordered[0]] = 1.0

        if ordered.size > 1:
            second_nearest[i] = symmetric_distance[
                i, ordered[1]
            ]
        else:
            second_nearest[i] = symmetric_distance[
                i, ordered[0]
            ]

    mutual_nearest = nearest * nearest.T

    mutual_top_two = np.zeros((n, n), dtype=float)
    top_two = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.where(valid[i])[0]
        if candidates.size == 0:
            continue

        ordered = candidates[
            np.argsort(symmetric_distance[i, candidates])
        ]
        top_two[i, ordered[: min(2, ordered.size)]] = 1.0

    mutual_top_two = top_two * top_two.T

    gap = np.maximum(second_nearest, global_scale)
    isolation_bonus = np.ones((n, n), dtype=float)

    for i in range(n):
        mask = valid[i]
        if np.any(mask):
            isolation_bonus[i, mask] += 0.25 * np.clip(
                gap[i] / np.maximum(
                    symmetric_distance[i, mask],
                    np.finfo(float).eps,
                ) - 1.0,
                0.0,
                3.0,
            )

    isolation_bonus = np.sqrt(
        isolation_bonus * isolation_bonus.T
    )

    scores = (
        bidirectional_preference
        * (0.65 + 0.35 * rank_continuity)
        * (1.0 + 0.75 * mutual_nearest + 0.25 * mutual_top_two)
        * isolation_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores
