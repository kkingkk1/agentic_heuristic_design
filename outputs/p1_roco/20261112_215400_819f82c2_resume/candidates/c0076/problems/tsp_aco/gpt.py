import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    edge_values = d[valid]

    if edge_values.size == 0:
        return np.zeros((n, n), dtype=float)

    low = float(np.percentile(edge_values, 10.0))
    high = float(np.percentile(edge_values, 90.0))

    if high - low <= eps:
        minimum = float(np.min(edge_values))
        maximum = float(np.max(edge_values))
        if maximum - minimum > eps:
            clipped = np.clip(d, minimum, maximum)
            distance_quality = 1.0 - (
                clipped - minimum
            ) / (maximum - minimum)
        else:
            distance_quality = np.ones((n, n), dtype=float)
    else:
        clipped = np.clip(d, low, high)
        distance_quality = 1.0 - (
            clipped - low
        ) / (high - low)

    distance_quality = np.where(valid, distance_quality, 0.0)
    np.fill_diagonal(distance_quality, 0.0)

    def normalize_valid(matrix: np.ndarray) -> np.ndarray:
        result = np.zeros_like(matrix, dtype=float)
        values = matrix[valid]
        if values.size == 0:
            return result
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum - minimum <= eps:
            result[valid] = 1.0
        else:
            result[valid] = (
                matrix[valid] - minimum
            ) / (maximum - minimum)
        return result

    global_quality = normalize_valid(distance_quality)

    flat = edge_values
    less = np.sum(flat[:, None] < flat[None, :], axis=1)
    equal = np.sum(flat[:, None] == flat[None, :], axis=1)
    global_rank = 1.0 - (
        less.astype(float) + 0.5 * equal.astype(float)
    ) / max(float(flat.size), 1.0)

    global_rank_matrix = np.zeros((n, n), dtype=float)
    global_rank_matrix[valid] = global_rank
    global_rank_matrix = normalize_valid(global_rank_matrix)

    local_rank_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        row_values = distance_quality[i, indices]

        less = np.sum(
            row_values[:, None] < row_values[None, :],
            axis=1,
        )
        equal = np.sum(
            row_values[:, None] == row_values[None, :],
            axis=1,
        )

        quality_rank = 1.0 - (
            less.astype(float) + 0.5 * equal.astype(float)
        ) / max(float(row_values.size), 1.0)

        local_rank_matrix[i, indices] = quality_rank

    local_rank_matrix = 0.5 * (
        local_rank_matrix + local_rank_matrix.T
    )
    local_rank_matrix = normalize_valid(local_rank_matrix)

    attractiveness = np.sum(distance_quality, axis=1)
    positive_attractiveness = attractiveness[attractiveness > eps]

    if positive_attractiveness.size:
        typical_attractiveness = float(
            np.median(positive_attractiveness)
        )
    else:
        typical_attractiveness = 1.0

    attractiveness = np.maximum(
        attractiveness,
        max(typical_attractiveness * 0.05, eps),
    )

    endpoint_balance = distance_quality / np.sqrt(
        attractiveness[:, None] * attractiveness[None, :]
    )
    endpoint_balance = normalize_valid(endpoint_balance)

    future_support = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        candidate_values = distance_quality[i, candidates]

        if candidates.size == 0:
            continue

        order = np.argsort(
            -candidate_values,
            kind="mergesort",
        )
        ordered_indices = candidates[order]
        ordered_values = candidate_values[order]

        for j in candidates:
            remaining = ordered_values[
                ordered_indices != j
            ]

            if remaining.size == 0:
                continuation = 0.0
            else:
                width = min(3, remaining.size)
                continuation = float(
                    np.mean(remaining[:width])
                )

            future_support[i, j] = continuation

    future_support = 0.5 * (
        future_support + future_support.T
    )
    future_support = normalize_valid(future_support)

    transition = distance_quality.copy()
    row_sums = np.sum(transition, axis=1)
    row_sums = np.maximum(row_sums, eps)
    transition = transition / row_sums[:, None]
    np.fill_diagonal(transition, 0.0)

    two_hop_support = transition @ transition.T
    two_hop_support = np.maximum(two_hop_support, 0.0)
    two_hop_support = normalize_valid(two_hop_support)

    score = (
        0.34 * global_quality
        + 0.18 * global_rank_matrix
        + 0.16 * local_rank_matrix
        + 0.15 * endpoint_balance
        + 0.13 * future_support
        + 0.04 * two_hop_support
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    values = score[valid]
    if values.size:
        minimum = float(np.min(values))
        maximum = float(np.max(values))

        if maximum - minimum <= eps:
            score[valid] = 1.0
        else:
            score[valid] = (
                score[valid] - minimum
            ) / (maximum - minimum)

        maximum = float(np.max(score))
        if maximum > eps:
            score /= maximum

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.asarray(score, dtype=float)
