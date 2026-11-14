import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, distance_matrix, np.inf)

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(safe, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)[:, None]

    denominator = float(max(n - 2, 1))
    row_percentile = np.clip(row_rank / denominator, 0.0, 1.0)
    col_percentile = np.clip(col_rank / denominator, 0.0, 1.0)

    # Low percentile means a strong edge from that endpoint.
    bilateral_strength = np.sqrt(
        (1.0 - row_percentile) * (1.0 - col_percentile)
    )

    # Measure how distinctive each edge is relative to the next available choice.
    row_sorted = np.sort(safe, axis=1)
    col_sorted = np.sort(safe, axis=0)

    best_row = row_sorted[:, 0]
    second_row = row_sorted[:, 1]
    best_col = col_sorted[0, :]
    second_col = col_sorted[1, :]

    row_gap = np.maximum(second_row - best_row, 0.0)
    col_gap = np.maximum(second_col - best_col, 0.0)

    row_scale = np.maximum(second_row, np.finfo(float).eps)
    col_scale = np.maximum(second_col, np.finfo(float).eps)

    row_exclusivity = np.clip(row_gap / row_scale, 0.0, 1.0)
    col_exclusivity = np.clip(col_gap / col_scale, 0.0, 1.0)

    edge_row_exclusivity = np.broadcast_to(
        row_exclusivity[:, None], (n, n)
    )
    edge_col_exclusivity = np.broadcast_to(
        col_exclusivity[None, :], (n, n)
    )

    exclusivity = np.sqrt(
        edge_row_exclusivity * edge_col_exclusivity
    )

    score = (
        bilateral_strength ** 2.2
        * (0.55 + 0.45 * exclusivity)
    )

    score[~valid] = 0.0
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
