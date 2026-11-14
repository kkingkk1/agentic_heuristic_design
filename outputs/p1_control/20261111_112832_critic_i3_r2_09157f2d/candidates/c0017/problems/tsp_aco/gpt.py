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

    d = np.maximum(distance_matrix.copy(), 0.0)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    finite_max = float(np.max(finite_values))
    large_value = max(finite_max, 1.0) * 10.0 + 1.0

    d[~np.isfinite(d)] = large_value
    np.fill_diagonal(d, large_value)

    off_diagonal_mask = ~np.eye(n, dtype=bool)
    off_diagonal = d[off_diagonal_mask]

    scale = float(np.median(off_diagonal))
    if not np.isfinite(scale) or scale <= 0.0:
        positive = off_diagonal[off_diagonal > 0.0]
        scale = float(np.mean(positive)) if positive.size else 1.0

    shortness = np.exp(-d / scale)

    row_rank_score = np.zeros_like(d, dtype=float)
    for i in range(n):
        valid = np.arange(n) != i
        local_distances = d[i, valid]
        order = np.argsort(local_distances, kind="stable")

        ranks = np.empty(n - 1, dtype=float)
        ranks[order] = np.arange(n - 1, dtype=float)

        denominator = max(n - 2, 1)
        row_rank_score[i, valid] = 1.0 - ranks / denominator

    nearest_indices = np.argmin(d, axis=1)
    mutual = np.zeros_like(d, dtype=float)

    for i in range(n):
        j = int(nearest_indices[i])
        if j != i and nearest_indices[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    row_sorted = np.sort(d, axis=1)
    nearest = row_sorted[:, 0]

    if n > 2:
        second_nearest = row_sorted[:, 1]
    else:
        second_nearest = nearest + scale

    gap = np.maximum(second_nearest - nearest, 0.0)
    competitiveness = np.zeros_like(d, dtype=float)

    for i in range(n):
        valid = np.arange(n) != i
        denominator = gap[i] + 0.1 * scale
        competitiveness[i, valid] = np.exp(
            -(d[i, valid] - nearest[i]) / denominator
        )

    mutual = 0.5 * (mutual + mutual.T)
    competitiveness = 0.5 * (competitiveness + competitiveness.T)

    base = 0.58 * shortness + 0.22 * row_rank_score

    structural_bonus = (
        0.20 * mutual
        + 0.18 * competitiveness
    )
    structural_bonus = np.minimum(structural_bonus, 0.50 * base)

    score = base + structural_bonus
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
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
