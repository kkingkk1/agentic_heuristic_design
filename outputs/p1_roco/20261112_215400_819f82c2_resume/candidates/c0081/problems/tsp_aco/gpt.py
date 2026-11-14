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
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    eps = np.finfo(float).eps
    valid = ~np.eye(n, dtype=bool)

    raw = np.maximum(distance_matrix, 0.0)
    np.fill_diagonal(raw, 0.0)

    symmetric = 0.5 * (raw + raw.T)
    off_raw = raw[valid]

    if off_raw.size == 0:
        return score

    asymmetry = np.abs(raw - raw.T)[valid]
    asym_scale = float(np.median(np.abs(off_raw))) + eps
    asym_ratio = float(np.median(asymmetry)) / asym_scale
    directional_weight = float(np.clip(asym_ratio, 0.0, 0.35))

    effective = (
        (1.0 - directional_weight) * symmetric
        + directional_weight * raw
    )
    np.fill_diagonal(effective, 0.0)

    positive = effective[valid]
    positive = positive[positive > 0.0]

    if positive.size:
        global_median = float(np.median(positive))
        global_mad = float(np.median(np.abs(positive - global_median)))
        global_scale = global_median + 1.4826 * global_mad
        if global_scale <= 0.0:
            global_scale = global_median
    else:
        global_scale = 1.0

    floor = max(global_scale * 1e-3, eps)

    def robust_scale(values: np.ndarray) -> float:
        if values.size == 0:
            return global_scale

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = median + 1.4826 * mad

        if scale <= floor:
            positive_values = values[values > 0.0]
            if positive_values.size:
                scale = float(np.median(positive_values))
            else:
                scale = global_scale

        return max(scale, floor)

    row_scales = np.empty(n, dtype=float)
    col_scales = np.empty(n, dtype=float)

    for i in range(n):
        row_scales[i] = robust_scale(effective[i, valid[i]])
        col_scales[i] = robust_scale(effective[valid[:, i], i])

    pair_scale = np.sqrt(
        np.maximum(row_scales[:, None], floor)
        * np.maximum(col_scales[None, :], floor)
    )
    pair_scale = np.maximum(pair_scale, floor)

    locality = np.zeros((n, n), dtype=float)
    locality[valid] = np.exp(-effective[valid] / pair_scale[valid])

    ranks = np.zeros((n, n), dtype=float)
    nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        values = effective[i, indices]

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

        minimum = float(np.min(values))
        nearest[i, indices] = (values == minimum).astype(float)

    scarcity = np.clip(1.0 - ranks, 0.0, 1.0)
    mutual_scarcity = np.sqrt(scarcity * scarcity.T)
    mutual_nearest = nearest * nearest.T

    zero_bonus = ((effective == 0.0) & valid).astype(float)

    # Explorer-inspired global quality signal with limited influence.
    edge_values = effective[valid]
    low = float(np.percentile(edge_values, 10.0))
    high = float(np.percentile(edge_values, 90.0))

    distance_quality = np.zeros((n, n), dtype=float)
    if high - low > eps:
        clipped = np.clip(effective, low, high)
        distance_quality[valid] = (
            (high - clipped[valid]) / (high - low)
        )
    else:
        minimum = float(np.min(edge_values))
        maximum = float(np.max(edge_values))
        if maximum - minimum > eps:
            distance_quality[valid] = (
                (maximum - edge_values) / (maximum - minimum)
            )
        else:
            distance_quality[valid] = 1.0

    distance_quality = np.clip(distance_quality, 0.0, 1.0)

    flat_quality = distance_quality[valid]
    order = np.argsort(-flat_quality, kind="mergesort")
    global_rank_values = np.zeros_like(flat_quality)

    if flat_quality.size == 1:
        global_rank_values[:] = 1.0
    else:
        global_rank_values[order] = (
            1.0
            - np.arange(flat_quality.size, dtype=float)
            / float(flat_quality.size - 1)
        )

    global_rank = np.zeros((n, n), dtype=float)
    global_rank[valid] = global_rank_values

    row_mass = np.maximum(np.sum(distance_quality, axis=1), eps)
    probabilities = distance_quality / row_mass[:, None]
    diversity = np.zeros(n, dtype=float)

    for i in range(n):
        values = probabilities[i, valid[i]]
        degree = values.size

        if degree <= 1:
            diversity[i] = 1.0
            continue

        positive_values = values[values > eps]
        if positive_values.size == 0:
            diversity[i] = 1.0
        else:
            entropy = -float(
                np.sum(positive_values * np.log(positive_values))
            )
            diversity[i] = entropy / np.log(float(degree))

    endpoint_diversity = np.sqrt(
        diversity[:, None] * diversity[None, :]
    )
    endpoint_diversity = np.where(valid, endpoint_diversity, 0.0)

    continuation = np.zeros((n, n), dtype=float)
    width = min(3, max(n - 2, 0))

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        values = distance_quality[i, indices]

        for position, j in enumerate(indices):
            alternatives = np.delete(values, position)

            if alternatives.size == 0:
                continuation[i, j] = 0.0
            else:
                k = min(width, alternatives.size)
                if k <= 0:
                    continuation[i, j] = float(np.mean(alternatives))
                else:
                    continuation[i, j] = float(
                        np.mean(np.sort(alternatives)[-k:])
                    )

    long_term = np.sqrt(
        np.maximum(continuation, 0.0)
        * np.maximum(continuation.T, 0.0)
    )
    long_term = np.where(valid, long_term, 0.0)

    # Main exploitation backbone; exploratory terms remain deliberately small.
    score = (
        0.62 * locality
        + 0.22 * mutual_scarcity
        + 0.08 * mutual_nearest
        + 0.02 * zero_bonus
        + 0.025 * distance_quality
        + 0.015 * global_rank
        + 0.01 * endpoint_diversity
        + 0.005 * long_term
    )
    score[~valid] = 0.0

    values = score[valid]
    low = float(np.min(values))
    high = float(np.max(values))

    if high - low > eps:
        score[valid] = (values - low) / (high - low)
    elif high > 0.0:
        score[valid] = 1.0

    np.fill_diagonal(score, 0.0)
    return np.clip(score, 0.0, 1.0)
