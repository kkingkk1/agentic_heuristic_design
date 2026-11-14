import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, long-horizon priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    values = distances[valid]
    positive_values = values[values > 0.0]

    if positive_values.size:
        positive_median = float(np.median(positive_values))
        positive_q25, positive_q75 = np.percentile(
            positive_values, [25.0, 75.0]
        )
        positive_iqr = max(
            float(positive_q75 - positive_q25),
            eps,
        )
        scale = max(positive_median, positive_iqr, eps)
    else:
        scale = 1.0

    q25, q75 = np.percentile(values, [25.0, 75.0])
    robust_iqr = max(float(q75 - q25), eps)

    # The upper bound is deliberately not replaced by max(values).
    clip_upper = max(
        float(q75 + 4.0 * max(robust_iqr, scale)),
        scale,
    )

    clipped = np.where(
        valid,
        np.minimum(np.maximum(distances, 0.0), clip_upper),
        0.0,
    )
    normalized = clipped / max(scale, eps)

    inverse_attraction = 1.0 / (1.0 + normalized)
    exponential_attraction = np.exp(
        -normalized / max(float(np.percentile(values, 75.0)) / scale, 1.0)
    )

    inverse_attraction[~valid] = 0.0
    exponential_attraction[~valid] = 0.0

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        order = indices[
            np.argsort(clipped[i, indices], kind="mergesort")
        ]

        if order.size == 1:
            row_rank[i, order[0]] = 1.0
        else:
            positions = np.arange(order.size, dtype=float)
            row_rank[i, order] = 1.0 - positions / (order.size - 1.0)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue

        order = indices[
            np.argsort(clipped[indices, j], kind="mergesort")
        ]

        if order.size == 1:
            col_rank[order[0], j] = 1.0
        else:
            positions = np.arange(order.size, dtype=float)
            col_rank[order, j] = 1.0 - positions / (order.size - 1.0)

    mutual_rank = np.sqrt(row_rank * col_rank)

    local_score = (
        0.46 * inverse_attraction
        + 0.22 * exponential_attraction
        + 0.16 * row_rank
        + 0.16 * col_rank
    )
    local_score[~valid] = 0.0

    degree = np.sum(valid, axis=1).astype(float)
    incoming_degree = np.sum(valid, axis=0).astype(float)

    # Effective alternative counts favor nodes with several reasonable
    # continuations while still assigning some value to sparse endpoints.
    row_affinity_mass = np.sum(inverse_attraction, axis=1)
    col_affinity_mass = np.sum(inverse_attraction, axis=0)

    row_top_mean = np.zeros(n, dtype=float)
    col_top_mean = np.zeros(n, dtype=float)

    for i in range(n):
        if degree[i] > 0:
            take = min(5, int(degree[i]))
            row_values = inverse_attraction[i, valid[i]]
            row_top_mean[i] = float(
                np.mean(np.partition(row_values, -take)[-take:])
            )

        if incoming_degree[i] > 0:
            take = min(5, int(incoming_degree[i]))
            col_values = inverse_attraction[valid[:, i], i]
            col_top_mean[i] = float(
                np.mean(np.partition(col_values, -take)[-take:])
            )

    row_diversity = row_top_mean * np.sqrt(
        np.maximum(degree, 1.0)
    ) / np.maximum(row_affinity_mass, eps)
    col_diversity = col_top_mean * np.sqrt(
        np.maximum(incoming_degree, 1.0)
    ) / np.maximum(col_affinity_mass, eps)

    if np.max(row_diversity) > 0.0:
        row_diversity /= np.max(row_diversity)
    if np.max(col_diversity) > 0.0:
        col_diversity /= np.max(col_diversity)

    future_bridge = np.sqrt(
        col_top_mean[:, None] * row_top_mean[None, :]
    )

    future_option = np.sqrt(
        col_diversity[:, None] * row_diversity[None, :]
    )

    long_horizon = (
        0.72 * future_bridge
        + 0.28 * future_bridge * future_option
    )
    long_horizon[~valid] = 0.0

    raw = (
        0.70 * local_score
        + 0.16 * mutual_rank
        + 0.14 * long_horizon
    )
    raw[~valid] = 0.0

    row_mass = np.sum(raw, axis=1, keepdims=True)
    col_mass = np.sum(raw, axis=0, keepdims=True)

    load_relief = 1.0 / np.sqrt(
        (row_mass + eps) * (col_mass + eps)
    )
    load_relief[~valid] = 0.0

    relief_max = float(np.max(load_relief))
    if relief_max > 0.0:
        load_relief /= relief_max

    balanced = raw * load_relief
    balanced_max = float(np.max(balanced))
    if balanced_max > 0.0:
        balanced /= balanced_max

    score = (
        0.78 * raw
        + 0.22 * balanced
    )
    score[~valid] = 0.0

    # Keep all legal edges, but soften the dominance of local winners.
    # Sparse rows are never gated.
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        degree_i = candidates.size

        if degree_i <= 1:
            continue

        candidate_count = min(
            degree_i,
            max(3, int(np.ceil(np.sqrt(degree_i))))
        )

        if degree_i <= candidate_count:
            continue

        order = candidates[
            np.argsort(-score[i, candidates], kind="mergesort")
        ]

        gate = np.full(n, 0.24, dtype=float)
        gate[order[:candidate_count]] = 1.0

        middle_start = candidate_count
        middle_end = min(
            degree_i,
            candidate_count + max(1, candidate_count // 2),
        )
        gate[order[middle_start:middle_end]] = 0.56

        # Give scarce incoming endpoints a larger exploration floor.
        rare_endpoint = load_relief[i] > 0.55
        gate[rare_endpoint & valid[i]] = np.maximum(
            gate[rare_endpoint & valid[i]],
            0.38,
        )

        score[i] *= gate

    score[~valid] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        result = score / maximum

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return result
