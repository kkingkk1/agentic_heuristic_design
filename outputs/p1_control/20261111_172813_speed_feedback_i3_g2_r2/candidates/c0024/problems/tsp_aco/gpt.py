import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    original = distance_matrix.copy()
    valid = np.isfinite(original)
    np.fill_diagonal(valid, False)

    positive = valid & (original > 0)
    if np.any(positive):
        fallback = float(np.median(original[positive]))
    else:
        fallback = 1.0

    work = original.copy()
    work[~valid] = fallback
    work = np.maximum(work, np.finfo(float).eps)
    np.fill_diagonal(work, np.inf)

    edge_mask = valid
    finite_positive = np.isfinite(work) & (work > 0)

    inverse = np.zeros_like(work, dtype=float)
    inverse[finite_positive] = 1.0 / work[finite_positive]

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    row_preference = inverse / row_max

    col_max = np.max(inverse, axis=0, keepdims=True)
    col_max[col_max <= 0] = 1.0
    col_preference = inverse / col_max

    bidirectional_preference = np.sqrt(
        np.maximum(row_preference, 0.0) *
        np.maximum(col_preference, 0.0)
    )

    k = max(1, min(5, n - 1))
    neighbor_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        order = np.argsort(work[i])
        selected = order[:k]
        neighbor_mask[i, selected] = True

    mutual_neighbor = neighbor_mask & neighbor_mask.T

    overlap = neighbor_mask.astype(float) @ neighbor_mask.astype(float).T
    overlap_scale = float(max(k, 1))
    neighborhood_compatibility = np.clip(overlap / overlap_scale, 0.0, 1.0)

    compatibility = (
        0.55 * mutual_neighbor.astype(float) +
        0.45 * neighborhood_compatibility
    )
    compatibility = np.clip(compatibility, 0.0, 1.0)

    best_two_step = np.full((n, n), np.inf, dtype=float)
    for mid in range(n):
        candidate = work[:, mid, None] + work[mid, None, :]
        best_two_step = np.minimum(best_two_step, candidate)

    np.fill_diagonal(best_two_step, np.inf)

    savings = np.zeros((n, n), dtype=float)
    usable = np.isfinite(best_two_step) & np.isfinite(work)
    savings[usable] = np.maximum(
        best_two_step[usable] - work[usable],
        0.0
    ) / np.maximum(best_two_step[usable], np.finfo(float).eps)

    score = bidirectional_preference
    score *= 0.72 + 0.28 * compatibility
    score *= 0.84 + 0.16 * savings

    score = 0.5 * (score + score.T)
    score[~edge_mask] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    score[~np.isfinite(score)] = 0.0
    return score.reshape(distance_matrix.shape)
