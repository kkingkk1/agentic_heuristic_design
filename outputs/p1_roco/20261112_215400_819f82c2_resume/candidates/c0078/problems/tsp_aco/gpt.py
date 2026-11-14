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
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    eps = np.finfo(float).eps
    valid = ~np.eye(n, dtype=bool)

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    def normalize(matrix: np.ndarray) -> np.ndarray:
        output = np.zeros_like(matrix, dtype=float)
        values = matrix[valid]
        if values.size == 0:
            return output

        low = float(np.min(values))
        high = float(np.max(values))

        if high - low <= eps:
            output[valid] = 1.0 if high > 0.0 else 0.0
        else:
            output[valid] = (values - low) / (high - low)

        return output

    edge_values = d[valid]
    low = float(np.percentile(edge_values, 10.0))
    high = float(np.percentile(edge_values, 90.0))

    if high - low <= eps:
        low = float(np.min(edge_values))
        high = float(np.max(edge_values))

    if high - low <= eps:
        distance_quality = np.ones((n, n), dtype=float)
        distance_quality[~valid] = 0.0
    else:
        clipped = np.clip(d, low, high)
        distance_quality = 1.0 - (clipped - low) / (high - low)
        distance_quality[~valid] = 0.0

    global_quality = normalize(distance_quality)

    positive = edge_values[edge_values > 0.0]
    if positive.size:
        global_median = float(np.median(positive))
        global_mad = float(np.median(np.abs(positive - global_median)))
        global_scale = global_median + 1.4826 * global_mad
        if global_scale <= eps:
            global_scale = global_median
    else:
        global_scale = 1.0

    floor = max(global_scale * 1e-3, eps)
    local_scales = np.empty(n, dtype=float)

    for i in range(n):
        row = d[i, valid[i]]
        row_positive = row[row > 0.0]

        if row_positive.size:
            row_median = float(np.median(row_positive))
            row_mad = float(np.median(np.abs(row_positive - row_median)))
            scale = row_median + 1.4826 * row_mad
            if scale <= eps:
                scale = row_median
        else:
            scale = global_scale

        local_scales[i] = max(scale, floor)

    pair_scale = np.sqrt(
        np.maximum(local_scales[:, None], floor)
        * np.maximum(local_scales[None, :], floor)
    )

    locality = np.zeros((n, n), dtype=float)
    locality[valid] = np.exp(-d[valid] / pair_scale[valid])

    ranks = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        values = d[i, indices]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]

        rank_values = np.empty(sorted_values.size, dtype=float)
        start = 0

        while start < sorted_values.size:
            end = start + 1
            while (
                end < sorted_values.size
                and sorted_values[end] == sorted_values[start]
            ):
                end += 1

            rank_values[start:end] = 0.5 * (start + end - 1)
            start = end

        row_ranks = np.empty_like(rank_values)
        row_ranks[order] = rank_values

        denominator = max(sorted_values.size - 1, 1)
        ranks[i, indices] = row_ranks / denominator

    scarcity = 1.0 - ranks
    mutual_scarcity = np.sqrt(
        np.clip(scarcity, 0.0, 1.0)
        * np.clip(scarcity.T, 0.0, 1.0)
    )

    nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        minimum = float(np.min(row))

        if np.isfinite(minimum):
            nearest[i] = (row == minimum).astype(float)
            nearest[i, i] = 0.0

    mutual_nearest = nearest * nearest.T

    attractiveness = np.sum(distance_quality, axis=1)
    positive_attractiveness = attractiveness[attractiveness > eps]

    if positive_attractiveness.size:
        typical = float(np.median(positive_attractiveness))
    else:
        typical = 1.0

    attractiveness = np.maximum(
        attractiveness,
        max(typical * 0.05, eps),
    )

    endpoint_balance = distance_quality / np.sqrt(
        attractiveness[:, None] * attractiveness[None, :]
    )
    endpoint_balance = normalize(endpoint_balance)

    future_support = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        values = distance_quality[i, indices]

        order = np.argsort(-values, kind="mergesort")
        ordered_indices = indices[order]
        ordered_values = values[order]

        for j in indices:
            remaining = ordered_values[ordered_indices != j]
            width = min(3, remaining.size)

            if width:
                future_support[i, j] = float(
                    np.mean(remaining[:width])
                )

    future_support = 0.5 * (
        future_support + future_support.T
    )
    future_support = normalize(future_support)

    transition = distance_quality.copy()
    row_sums = np.sum(transition, axis=1)
    row_sums = np.maximum(row_sums, eps)
    transition = transition / row_sums[:, None]
    transition[~valid] = 0.0

    two_hop_support = transition @ transition.T
    two_hop_support = normalize(two_hop_support)

    score = (
        0.56 * locality
        + 0.18 * mutual_scarcity
        + 0.08 * mutual_nearest
        + 0.08 * global_quality
        + 0.05 * endpoint_balance
        + 0.03 * future_support
        + 0.02 * two_hop_support
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    score = np.maximum(score, 0.0)

    values = score[valid]
    if values.size:
        low = float(np.min(values))
        high = float(np.max(values))

        if high - low <= eps:
            score[valid] = 1.0 if high > 0.0 else 0.0
        else:
            score[valid] = (values - low) / (high - low)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    return np.asarray(np.clip(score, 0.0, 1.0), dtype=float)
