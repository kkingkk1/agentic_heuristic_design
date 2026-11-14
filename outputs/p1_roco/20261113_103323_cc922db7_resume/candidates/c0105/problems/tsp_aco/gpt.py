import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric distance- and neighborhood-aware TSP edge priors."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(raw)
    nonnegative = raw >= 0.0
    valid_raw = finite & nonnegative
    np.fill_diagonal(valid_raw, False)

    positive = raw[valid_raw & (raw > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = raw[valid_raw]
        global_scale = (
            float(np.median(finite_values))
            if finite_values.size
            else 1.0
        )

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    # Construct a symmetric distance view while preserving one-sided finite edges.
    d = np.maximum(raw, 0.0)
    symmetric = np.full((n, n), np.inf, dtype=float)

    both = valid_raw & valid_raw.T
    symmetric[both] = 0.5 * (d[both] + d.T[both])

    forward_only = valid_raw & ~valid_raw.T
    backward_only = valid_raw.T & ~valid_raw
    symmetric[forward_only] = d[forward_only]
    symmetric[backward_only] = d.T[backward_only]

    np.fill_diagonal(symmetric, np.inf)
    valid = np.isfinite(symmetric)

    if not np.any(valid):
        return scores

    # Robust local scale: combine the nearest and median finite neighborhood distances.
    local_scale = np.full(n, global_scale, dtype=float)
    for i in range(n):
        neighbors = symmetric[i, valid[i]]
        if neighbors.size:
            nearest = float(np.min(neighbors))
            median = float(np.median(neighbors))
            local_scale[i] = 0.5 * nearest + 0.5 * median

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )
    normalized = np.divide(
        symmetric,
        pair_scale,
        out=np.full_like(symmetric, np.inf),
        where=valid,
    )

    # Soft distance preference, adaptive to both endpoints.
    distance_preference = np.exp(-np.clip(normalized, 0.0, 60.0))

    # Rank consistency from both endpoints.
    rank_score = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[np.argsort(symmetric[i, candidates], kind="stable")]
            rank_score[i, order] = 1.0 / (
                1.0 + np.arange(order.size, dtype=float)
            )

    bidirectional_rank = np.sqrt(rank_score * rank_score.T)

    # Endpoint-adaptive softmax preference.
    preference = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            logits = -np.clip(normalized[i, candidates], 0.0, 60.0)
            logits -= np.max(logits)
            weights = np.exp(logits)
            total = float(np.sum(weights))
            if total > 0.0 and np.isfinite(total):
                preference[i, candidates] = weights / total

    bidirectional_preference = np.sqrt(preference * preference.T)

    # Mutual nearest-neighbor structure.
    nearest = np.full(n, -1, dtype=int)
    second_distance = np.full(n, global_scale, dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[np.argsort(symmetric[i, candidates], kind="stable")]
            nearest[i] = int(order[0])
            if order.size > 1:
                second_distance[i] = float(symmetric[i, order[1]])
            else:
                second_distance[i] = float(symmetric[i, order[0]])

    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Scarcity increases when the nearest edge is clearly better than alternatives.
    nearest_distance = np.full(n, global_scale, dtype=float)
    for i in range(n):
        if nearest[i] >= 0:
            nearest_distance[i] = symmetric[i, nearest[i]]

    gap_ratio = second_distance / np.maximum(nearest_distance, eps)
    gap_ratio = np.clip(gap_ratio - 1.0, 0.0, 4.0)

    scarcity = np.ones((n, n), dtype=float)
    for i in range(n):
        if np.any(valid[i]):
            edge_ratio = second_distance[i] / np.maximum(symmetric[i], eps)
            scarcity[i, valid[i]] += 0.20 * np.clip(
                edge_ratio - 1.0,
                0.0,
                4.0,
            )
            scarcity[i, valid[i]] += 0.10 * gap_ratio[i]

    bidirectional_scarcity = np.sqrt(scarcity * scarcity.T)

    scores = (
        distance_preference
        * (0.55 + 0.45 * bidirectional_preference)
        * (0.60 + 0.40 * bidirectional_rank)
        * (1.0 + 0.55 * mutual)
        * bidirectional_scarcity
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float)
