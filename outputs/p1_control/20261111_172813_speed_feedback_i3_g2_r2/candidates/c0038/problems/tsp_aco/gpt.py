import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
        & ~diagonal
    )

    positive = valid & (distance_matrix > 0.0)
    fallback = (
        float(np.median(distance_matrix[positive]))
        if np.any(positive)
        else 1.0
    )

    distances = np.where(valid, distance_matrix, fallback)
    distances = np.maximum(distances, np.finfo(float).eps)
    distances[diagonal] = np.inf

    inverse = np.divide(
        1.0,
        distances,
        out=np.zeros_like(distances),
        where=np.isfinite(distances) & (distances > 0.0),
    )

    zero_edges = valid & (distance_matrix == 0.0)
    local = np.zeros_like(inverse)

    rows_with_zero = np.any(zero_edges, axis=1)
    nonzero_rows = np.flatnonzero(~rows_with_zero)
    zero_rows = np.flatnonzero(rows_with_zero)

    if nonzero_rows.size:
        row_max = np.max(inverse[nonzero_rows], axis=1, keepdims=True)
        row_max = np.maximum(row_max, 1.0)
        local[nonzero_rows] = inverse[nonzero_rows] / row_max

    if zero_rows.size:
        local[zero_edges] = 1.0

    top_k = min(n - 1, max(4, int(np.sqrt(n)) + 1))
    tail_weight = 1.0 / np.log2(top_k + 2.0)

    weighted = local * tail_weight
    weighted[~valid] = 0.0

    candidate_values = np.where(valid, -local, np.inf)
    top_indices = np.argpartition(
        candidate_values,
        kth=top_k - 1,
        axis=1,
    )[:, :top_k]

    rows = np.arange(n)[:, None]
    selected_local = local[rows, top_indices]

    order = np.argsort(
        -selected_local,
        axis=1,
        kind="stable",
    )
    ranked_indices = np.take_along_axis(top_indices, order, axis=1)
    ranked_local = np.take_along_axis(selected_local, order, axis=1)

    ranks = np.arange(1, top_k + 1, dtype=float)[None, :]
    rank_weights = 1.0 / np.log2(ranks + 1.0)

    selected_valid = valid[rows, ranked_indices]
    selected_weighted = ranked_local * rank_weights
    selected_weighted = np.where(
        selected_valid,
        selected_weighted,
        0.0,
    )

    weighted[rows, ranked_indices] = selected_weighted

    score = np.sqrt(
        np.maximum(weighted, 0.0)
        * np.maximum(weighted.T, 0.0)
    )
    score = 0.5 * (score + score.T)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
