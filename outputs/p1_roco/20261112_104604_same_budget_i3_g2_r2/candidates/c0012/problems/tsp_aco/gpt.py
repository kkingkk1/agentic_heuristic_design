import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(distance_matrix)
    valid = finite & (distance_matrix >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    symmetric_valid = valid | valid.T
    effective = np.full((n, n), np.inf, dtype=float)

    both = valid & valid.T
    only_forward = valid & ~valid.T
    only_reverse = valid.T & ~valid

    effective[both] = (
        distance_matrix[both] + distance_matrix.T[both]
    ) * 0.5
    effective[only_forward] = distance_matrix[only_forward]
    effective[only_reverse] = distance_matrix.T[only_reverse]
    np.fill_diagonal(effective, np.inf)

    positive = effective[symmetric_valid & (effective > 0)]
    if positive.size:
        global_scale = float(np.median(positive))
        q25, q75 = np.percentile(positive, [25.0, 75.0])
        robust_spread = max(float(q75 - q25), np.finfo(float).eps)
    else:
        global_scale = 1.0
        robust_spread = 1.0

    row_scale = np.full((n, 1), global_scale, dtype=float)
    for i in range(n):
        values = effective[i, symmetric_valid[i] & (effective[i] > 0)]
        if values.size:
            row_scale[i, 0] = float(np.median(values))

    scale = 0.5 * (global_scale + row_scale)
    scale = np.maximum(scale, np.finfo(float).eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[symmetric_valid] = np.exp(
        -effective[symmetric_valid] / scale[symmetric_valid[:, 0]]
    )
    np.fill_diagonal(affinity, 0.0)

    robust_distance = np.zeros_like(effective)
    robust_distance[symmetric_valid] = (
        effective[symmetric_valid] / (global_scale + robust_spread)
    )

    local_affinity = np.zeros_like(affinity)
    for i in range(n):
        mask = symmetric_valid[i]
        if np.any(mask):
            values = robust_distance[i, mask]
            order = np.argsort(values, kind="stable")
            ranked = np.zeros(values.size, dtype=float)
            if values.size > 1:
                ranked[order] = 1.0 - (
                    np.arange(values.size, dtype=float) / (values.size - 1.0)
                )
            else:
                ranked[order] = 1.0
            local_affinity[i, mask] = ranked

    row_max = np.max(affinity, axis=1, keepdims=True)
    normalized = affinity / np.maximum(row_max, np.finfo(float).eps)

    mutual = np.sqrt(np.maximum(normalized * normalized.T, 0.0))
    mutual *= symmetric_valid

    continuation_two = normalized @ normalized
    continuation_three = continuation_two @ normalized
    long_term = 0.6 * continuation_two.T + 0.4 * continuation_three.T

    long_term_max = np.max(long_term, axis=1, keepdims=True)
    long_term = long_term / np.maximum(
        long_term_max, np.finfo(float).eps
    )

    neighborhood_overlap = normalized @ normalized.T
    neighborhood_mass = np.sqrt(
        np.maximum(
            np.sum(normalized * normalized, axis=1, keepdims=True)
            * np.sum(normalized * normalized, axis=1, keepdims=True).T,
            np.finfo(float).eps,
        )
    )
    overlap_ratio = neighborhood_overlap / neighborhood_mass
    diversity = np.clip(1.0 - overlap_ratio, 0.0, 1.0)
    diversity *= symmetric_valid

    savings = np.zeros_like(affinity)
    for i in range(n):
        mask = symmetric_valid[i]
        values = effective[i, mask]
        if values.size >= 2:
            second = float(np.partition(values, 1)[1])
            savings[i, mask] = np.maximum(
                (second - values) / (second + values + np.finfo(float).eps),
                0.0,
            )

    score = (
        0.28 * normalized
        + 0.20 * local_affinity
        + 0.17 * mutual
        + 0.22 * long_term
        + 0.08 * diversity
        + 0.05 * savings
    )

    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
