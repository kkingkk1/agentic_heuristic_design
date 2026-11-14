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

    valid = ~np.eye(n, dtype=bool)

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    off_diagonal = d[valid]
    positive = off_diagonal[off_diagonal > 0.0]

    if positive.size:
        global_median = float(np.median(positive))
        global_mad = float(
            np.median(np.abs(positive - global_median))
        )
        global_scale = global_median + 1.4826 * global_mad
        if global_scale <= eps:
            global_scale = max(global_median, eps)
    else:
        global_scale = 1.0

    floor = max(global_scale * 1e-3, eps)
    local_scales = np.empty(n, dtype=float)

    for i in range(n):
        row = d[i, np.arange(n) != i]

        row_median = float(np.median(row))
        row_mad = float(np.median(np.abs(row - row_median)))
        scale = row_median + 1.4826 * row_mad

        if scale <= floor:
            positive_row = row[row > 0.0]
            if positive_row.size:
                scale = float(np.median(positive_row))
            else:
                scale = global_scale

        local_scales[i] = max(scale, floor)

    pair_scale = np.sqrt(
        np.maximum(local_scales[:, None], floor)
        * np.maximum(local_scales[None, :], floor)
    )
    pair_scale = np.maximum(pair_scale, floor)

    locality = np.zeros((n, n), dtype=float)
    locality[valid] = np.exp(-d[valid] / pair_scale[valid])

    global_locality = np.zeros((n, n), dtype=float)
    global_locality[valid] = np.exp(-d[valid] / global_scale)

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

    scarcity = 1.0 - np.clip(ranks, 0.0, 1.0)
    mutual_scarcity = np.sqrt(
        np.clip(scarcity, 0.0, 1.0)
        * np.clip(scarcity.T, 0.0, 1.0)
    )

    nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        row = d[i].copy()
        row[i] = np.inf
        minimum = np.min(row)

        if np.isfinite(minimum):
            nearest[i] = (row == minimum).astype(float)
            nearest[i, i] = 0.0

    mutual_nearest = nearest * nearest.T
    zero_bonus = ((d == 0.0) & valid).astype(float)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-d[valid] / global_scale)

    row_mass = affinity.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)
    normalized_affinity = affinity / row_mass[:, None]

    common_neighbor = normalized_affinity @ normalized_affinity.T
    common_neighbor = np.maximum(common_neighbor, 0.0)
    common_max = float(np.max(common_neighbor))
    if common_max > 0.0:
        common_neighbor /= common_max

    endpoint_opportunity = np.sqrt(
        np.maximum(row_mass[:, None], 0.0)
        * np.maximum(row_mass[None, :], 0.0)
    )
    opportunity_max = float(np.max(endpoint_opportunity))
    if opportunity_max > 0.0:
        endpoint_opportunity /= opportunity_max

    score = (
        0.50 * locality
        + 0.15 * global_locality
        + 0.15 * mutual_scarcity
        + 0.10 * mutual_nearest
        + 0.07 * common_neighbor
        + 0.02 * endpoint_opportunity
        + 0.01 * zero_bonus
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return np.clip(score, 0.0, 1.0)
