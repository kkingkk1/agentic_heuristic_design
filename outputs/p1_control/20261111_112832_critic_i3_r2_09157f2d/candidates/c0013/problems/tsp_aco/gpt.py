import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")

    n, m = distance_matrix.shape
    if n != m:
        raise ValueError("distance_matrix must be a square 2D array")
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

    off_diagonal = d[~np.eye(n, dtype=bool)]
    scale = np.median(off_diagonal)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = np.mean(off_diagonal)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    shortness = np.exp(-d / scale)

    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.broadcast_to(
        np.arange(n, dtype=float),
        (n, n),
    )
    np.put_along_axis(ranks, order, rank_values, axis=1)

    row_rank_score = 1.0 - ranks / max(n - 2, 1)
    np.fill_diagonal(row_rank_score, 0.0)

    nearest_indices = order[:, 0]
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    mutual = (
        (nearest_indices[:, None] == cols)
        & (nearest_indices[None, :] == rows)
    ).astype(float)

    row_sorted = np.take_along_axis(d, order, axis=1)
    nearest = row_sorted[:, 0]
    second_nearest = (
        row_sorted[:, 1]
        if n > 2
        else nearest + scale
    )

    gap = np.maximum(second_nearest - nearest, 0.0)
    denominator = gap + scale * 0.1
    competitiveness = np.exp(
        -(d - nearest[:, None]) / denominator[:, None]
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
