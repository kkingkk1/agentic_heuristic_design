import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0) & ~diagonal
    if not np.any(finite_nonnegative):
        return result

    positive_values = d[finite_nonnegative & (d > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        finite_values = d[finite_nonnegative]
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    forward = np.where(finite_nonnegative, d, np.nan)
    backward = forward.T

    forward_valid = np.isfinite(forward)
    backward_valid = np.isfinite(backward)

    both_valid = forward_valid & backward_valid
    only_forward = forward_valid & ~backward_valid
    only_backward = ~forward_valid & backward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = (
        forward[both_valid] + backward[both_valid]
    ) / 2.0
    distance[only_forward] = forward[only_forward]
    distance[only_backward] = backward[only_backward]

    valid = np.isfinite(distance) & ~diagonal
    np.fill_diagonal(distance, np.inf)

    if not np.any(valid):
        return result

    order = np.argsort(distance, axis=1, kind="stable")
    local_k = min(5, n - 1)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    valid_count = np.sum(valid, axis=1).astype(float)

    for i in range(n):
        neighbors = order[i][valid[i, order[i]]]
        if neighbors.size == 0:
            continue

        values = distance[i, neighbors]

        positive = values[values > 0.0]
        scale_values = positive[:local_k] if positive.size else values[:local_k]
        if scale_values.size:
            local_scale[i] = float(np.median(scale_values))

        if values.size >= 2:
            second_distance[i] = float(values[1])
        else:
            second_distance[i] = float(values[0])

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)

    pair_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            eps,
        )
    )

    normalized_distance = np.divide(
        distance,
        pair_scale,
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 50.0)

    shortness = np.exp(-0.85 * normalized_distance)
    shortness[~valid] = 0.0

    row_rank = np.full((n, n), float(n), dtype=float)
    for i in range(n):
        neighbors = order[i][valid[i, order[i]]]
        row_rank[i, neighbors] = np.arange(
            neighbors.size,
            dtype=float,
        )

    rank_strength = 1.0 / np.power(1.0 + row_rank, 0.85)
    rank_consensus = np.sqrt(rank_strength * rank_strength.T)
    rank_consensus[~valid] = 0.0

    nearest = np.full(n, -1, dtype=int)
    top_k = min(3, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)

    for i in range(n):
        neighbors = order[i][valid[i, order[i]]]
        if neighbors.size:
            nearest[i] = int(neighbors[0])
            top_neighbors[i, neighbors[:top_k]] = True

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    row_preference = np.zeros((n, n), dtype=float)
    for i in range(n):
        mask = valid[i]
        if not np.any(mask):
            continue

        logits = -normalized_distance[i, mask]
        logits -= np.max(logits)

        weights = np.exp(np.clip(logits, -50.0, 0.0))
        total = float(np.sum(weights))

        if np.isfinite(total) and total > 0.0:
            row_preference[i, mask] = weights / total

    mutual_preference = np.sqrt(
        row_preference * row_preference.T
    )
    endpoint_support = np.maximum(
        row_preference,
        row_preference.T,
    )

    scarcity = np.ones((n, n), dtype=float)
    for i in range(n):
        mask = valid[i]
        if not np.any(mask):
            continue

        gap = max(second_distance[i], global_scale)
        ratio = gap / np.maximum(distance[i, mask], eps)
        scarcity[i, mask] += 0.28 * np.clip(ratio - 1.0, 0.0, 4.0)

    scarcity = np.sqrt(scarcity * scarcity.T)
    scarcity = np.clip(scarcity, 1.0, 2.0)

    density = np.sqrt(
        np.clip(valid_count[:, None], 1.0, None)
        * np.clip(valid_count[None, :], 1.0, None)
    )
    density = np.clip(
        density / max(float(n - 1), 1.0),
        0.35,
        1.0,
    )

    base = (
        0.52 * shortness
        + 0.28 * rank_consensus
        + 0.20 * mutual_preference
    )

    structural_bonus = (
        1.0
        + 0.34 * mutual_nearest
        + 0.16 * mutual_top
        + 0.18 * endpoint_support
    )

    scores = (
        base
        * structural_bonus
        * (0.88 + 0.12 * scarcity)
        * (0.90 + 0.10 * density)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
