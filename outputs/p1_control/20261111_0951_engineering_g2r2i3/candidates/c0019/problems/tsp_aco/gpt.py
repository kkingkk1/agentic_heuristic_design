import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")

    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError("distance_matrix must be square")

    n = n_rows
    heuristic = np.zeros((n, n), dtype=float)

    if n <= 1:
        return heuristic

    valid = np.isfinite(distance_matrix) & (distance_matrix > 0)
    valid &= ~np.eye(n, dtype=bool)

    inverse_distance = np.zeros_like(distance_matrix)
    np.divide(
        1.0,
        distance_matrix,
        out=inverse_distance,
        where=valid,
    )

    row_scale = inverse_distance.max(axis=1, keepdims=True)
    local_score = np.zeros_like(inverse_distance)
    np.divide(
        inverse_distance,
        row_scale,
        out=local_score,
        where=row_scale > 0,
    )

    ranks = np.zeros_like(distance_matrix, dtype=float)
    valid_rows = valid.any(axis=1)

    if np.any(valid_rows):
        masked_distance = np.where(valid, distance_matrix, np.inf)
        order = np.argsort(masked_distance, axis=1, kind="stable")
        sorted_ranks = np.argsort(order, axis=1, kind="stable").astype(float)
        ranks[valid_rows] = sorted_ranks[valid_rows]

    rank_score = np.zeros_like(distance_matrix, dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])

    mutual_rank = np.sqrt(rank_score * rank_score.T)
    row_max = mutual_rank.max(axis=1, keepdims=True)
    normalized_mutual_rank = np.zeros_like(mutual_rank)
    np.divide(
        mutual_rank,
        row_max,
        out=normalized_mutual_rank,
        where=row_max > 0,
    )

    heuristic = 0.65 * local_score + 0.35 * normalized_mutual_rank
    heuristic = 0.5 * (heuristic + heuristic.T)

    heuristic[~np.isfinite(heuristic)] = 0.0
    heuristic = np.maximum(heuristic, 0.0)
    np.fill_diagonal(heuristic, 0.0)

    return heuristic
