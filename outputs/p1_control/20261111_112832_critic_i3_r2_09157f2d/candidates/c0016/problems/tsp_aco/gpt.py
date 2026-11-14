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

    d = np.maximum(distance_matrix.copy(), 0.0)
    finite_values = d[np.isfinite(d)]

    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    finite_max = float(np.max(finite_values))
    large_value = max(finite_max, 1.0) * 10.0 + 1.0
    d[~np.isfinite(d)] = large_value
    np.fill_diagonal(d, large_value)

    upper_rows, upper_cols = np.triu_indices(n, k=1)
    upper_distances = d[upper_rows, upper_cols]

    scale = float(np.median(upper_distances))
    if not np.isfinite(scale) or scale <= 0.0:
        positive = upper_distances[upper_distances > 0.0]
        scale = float(np.mean(positive)) if positive.size else 1.0

    nearest_two = np.argpartition(d, kth=1, axis=1)[:, :2]
    nearest_two_distances = np.take_along_axis(d, nearest_two, axis=1)

    nearest_order = np.argsort(nearest_two_distances, axis=1)
    nearest_indices = np.take_along_axis(nearest_two, nearest_order, axis=1)
    nearest_values = np.take_along_axis(
        nearest_two_distances,
        nearest_order,
        axis=1,
    )

    nearest = nearest_values[:, 0]
    second_nearest = nearest_values[:, 1]
    gap = np.maximum(second_nearest - nearest, 0.0)
    denominator = gap + 0.1 * scale

    row_rank = np.zeros((n, n), dtype=float)
    for i in range(n):
        valid = np.arange(n) != i
        nodes = np.flatnonzero(valid)
        order = np.argsort(d[i, valid], kind="stable")
        ranks = np.empty(n - 1, dtype=float)
        ranks[order] = np.arange(n - 1, dtype=float)
        row_rank[i, nodes] = 1.0 - ranks / max(n - 2, 1)

    endpoint_rank = 0.5 * (
        row_rank[upper_rows, upper_cols]
        + row_rank[upper_cols, upper_rows]
    )

    row_competition = np.exp(
        -(upper_distances - nearest[upper_rows])
        / denominator[upper_rows]
    )
    col_competition = np.exp(
        -(upper_distances - nearest[upper_cols])
        / denominator[upper_cols]
    )
    competitiveness = 0.5 * (row_competition + col_competition)

    mutual = (
        (nearest_indices[upper_rows, 0] == upper_cols)
        & (nearest_indices[upper_cols, 0] == upper_rows)
    ).astype(float)

    shortness = np.exp(-upper_distances / scale)
    base = 0.68 * shortness + 0.32 * endpoint_rank

    structural_bonus = 0.16 * mutual + 0.14 * competitiveness
    structural_bonus = np.minimum(structural_bonus, 0.50 * base)

    upper_score = base + structural_bonus

    score = np.zeros((n, n), dtype=float)
    score[upper_rows, upper_cols] = upper_score
    score[upper_cols, upper_rows] = upper_score

    row_max = np.max(score, axis=1, keepdims=True)
    np.divide(
        score,
        row_max,
        out=score,
        where=row_max > 0.0,
    )

    np.fill_diagonal(score, 0.0)
    return score
