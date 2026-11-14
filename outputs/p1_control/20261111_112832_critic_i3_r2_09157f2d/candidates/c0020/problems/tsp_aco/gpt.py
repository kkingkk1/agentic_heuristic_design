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
    finite = np.isfinite(d)

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    finite_values = d[finite]
    large_value = max(float(np.max(finite_values)), 1.0) * 10.0 + 1.0
    d[~finite] = large_value
    np.fill_diagonal(d, large_value)

    off_diagonal_mask = ~np.eye(n, dtype=bool)
    off_diagonal = d[off_diagonal_mask]

    scale = float(np.median(off_diagonal))
    if not np.isfinite(scale) or scale <= 0.0:
        positive = off_diagonal[off_diagonal > 0.0]
        scale = float(np.mean(positive)) if positive.size else 1.0

    epsilon = max(scale * 1e-12, np.finfo(float).eps)

    shortness = np.exp(-d / scale)
    reciprocal = 1.0 / (d + epsilon)
    reciprocal = 0.5 * (reciprocal + reciprocal.T)

    reciprocal_row_max = np.max(reciprocal, axis=1, keepdims=True)
    reciprocal = np.divide(
        reciprocal,
        reciprocal_row_max,
        out=np.zeros_like(reciprocal),
        where=reciprocal_row_max > 0.0,
    )

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(d, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)[None, :]

    rank_denominator = max(n - 2, 1)
    row_rank_score = 1.0 - ranks / rank_denominator
    row_rank_score = np.clip(row_rank_score, 0.0, 1.0)
    np.fill_diagonal(row_rank_score, 0.0)

    nearest_indices = np.argmin(d, axis=1)
    nearest_one_hot = np.eye(n, dtype=float)[nearest_indices]
    mutual = nearest_one_hot * nearest_one_hot.T

    sorted_distances = np.sort(d, axis=1)
    nearest = sorted_distances[:, 0]

    if n > 2:
        second_nearest = sorted_distances[:, 1]
    else:
        second_nearest = nearest + scale

    gap = np.maximum(second_nearest - nearest, 0.0)
    denominator = gap[:, None] + 0.1 * scale

    competitiveness = np.exp(
        -(d - nearest[:, None]) / denominator
    )
    np.fill_diagonal(competitiveness, 0.0)

    shortness = 0.5 * (shortness + shortness.T)
    row_rank_score = 0.5 * (row_rank_score + row_rank_score.T)
    competitiveness = 0.5 * (competitiveness + competitiveness.T)

    base = (
        0.40 * shortness
        + 0.25 * reciprocal
        + 0.20 * row_rank_score
    )

    structural_bonus = (
        0.06 * mutual
        + 0.09 * competitiveness
    )
    structural_bonus = np.minimum(structural_bonus, 0.25 * base)

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

    score = np.power(score, 0.85)
    np.fill_diagonal(score, 0.0)

    return score
