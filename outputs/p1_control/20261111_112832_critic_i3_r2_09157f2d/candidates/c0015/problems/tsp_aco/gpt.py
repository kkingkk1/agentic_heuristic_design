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
    finite = np.isfinite(d)
    finite_values = d[finite]

    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    max_finite = max(float(np.max(finite_values)), 0.0)
    large_value = max_finite * 10.0 + 1.0
    d[~finite] = large_value
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, large_value)

    upper_rows, upper_cols = np.triu_indices(n, k=1)
    upper_distances = d[upper_rows, upper_cols]

    scale = float(np.median(upper_distances))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(upper_distances))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    nearest_two = np.argpartition(d, kth=1, axis=1)[:, :2]
    nearest_two_distances = np.take_along_axis(d, nearest_two, axis=1)

    swap = nearest_two_distances[:, 1] < nearest_two_distances[:, 0]
    nearest_indices = np.where(
        swap,
        nearest_two[:, 1],
        nearest_two[:, 0],
    )
    nearest = np.minimum(
        nearest_two_distances[:, 0],
        nearest_two_distances[:, 1],
    )
    second_nearest = np.maximum(
        nearest_two_distances[:, 0],
        nearest_two_distances[:, 1],
    )

    gap = np.maximum(second_nearest - nearest, 0.0)
    denominator = gap + 0.1 * scale

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
        (nearest_indices[upper_rows] == upper_cols)
        & (nearest_indices[upper_cols] == upper_rows)
    ).astype(float)

    shortness = np.exp(-upper_distances / scale)
    upper_score = (
        0.55 * shortness
        + 0.30 * competitiveness
        + 0.15 * mutual
    )

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
