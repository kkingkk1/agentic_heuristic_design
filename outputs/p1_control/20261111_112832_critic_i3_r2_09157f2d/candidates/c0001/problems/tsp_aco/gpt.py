import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    large_value = np.max(finite_values) * 10.0 + 1.0
    d[~np.isfinite(d)] = large_value
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, large_value)

    scale = np.median(d[d < large_value])
    if not np.isfinite(scale) or scale <= 0.0:
        scale = np.mean(d[d < large_value])
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    shortness = np.exp(-d / scale)

    row_sorted = np.sort(d, axis=1)
    nearest = row_sorted[:, 0]
    second_nearest = row_sorted[:, 1] if n > 2 else nearest + scale

    row_rank_score = np.zeros_like(d)
    for i in range(n):
        valid = np.arange(n) != i
        local = d[i, valid]
        order = np.argsort(local)
        ranks = np.empty(n - 1, dtype=float)
        ranks[order] = np.arange(n - 1, dtype=float)
        row_rank_score[i, valid] = 1.0 - ranks / max(n - 2, 1)

    mutual = np.zeros_like(d)
    nearest_indices = np.argmin(d, axis=1)
    for i in range(n):
        j = nearest_indices[i]
        if j != i and nearest_indices[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    gap_i = np.maximum(second_nearest - nearest, 0.0)
    competitiveness = np.zeros_like(d)
    for i in range(n):
        valid = np.arange(n) != i
        denominator = gap_i[i] + scale * 0.1
        competitiveness[i, valid] = np.exp(
            -(d[i, valid] - nearest[i]) / denominator
        )

    score = (
        0.50 * shortness
        + 0.20 * row_rank_score
        + 0.15 * competitiveness
        + 0.15 * mutual
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    np.fill_diagonal(score, 0.0)
    return score
