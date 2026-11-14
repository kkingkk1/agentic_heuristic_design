import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distance_matrix)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.zeros_like(distance_matrix, dtype=float)
    distances[finite] = np.maximum(distance_matrix[finite], 0.0)
    np.fill_diagonal(distances, 0.0)

    valid_values = distances[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        scale = float(np.percentile(positive_values, 50.0))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.mean(positive_values))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    normalized = distances / scale
    offset = 0.35
    exponent = 1.35

    desirability = np.zeros_like(distances, dtype=float)
    desirability[valid] = np.power(
        1.0 / (offset + normalized[valid]),
        exponent,
    )

    row_best = np.full(n, scale, dtype=float)
    for i in range(n):
        row_values = distances[i, valid[i]]
        if row_values.size:
            row_best[i] = max(float(np.min(row_values)), 0.0)

    row_context = 0.25 * scale + row_best
    source_ratio = np.zeros_like(distances, dtype=float)
    target_ratio = np.zeros_like(distances, dtype=float)

    source_ratio[valid] = row_context[:, None][valid] / (
        distances[valid] + 0.25 * scale
    )
    target_ratio[valid] = row_context[None, :][valid] / (
        distances[valid] + 0.25 * scale
    )

    local_factor = np.sqrt(
        np.clip(source_ratio, 0.0, 1.0)
        * np.clip(target_ratio, 0.0, 1.0)
    )

    scores = desirability * (0.72 + 0.28 * local_factor)

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            nearest[i] = candidates[np.argmin(distances[i, candidates])]

    mutual = np.zeros_like(scores, dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0

    scores *= 1.0 + 0.15 * mutual
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.reshape(n, n)
