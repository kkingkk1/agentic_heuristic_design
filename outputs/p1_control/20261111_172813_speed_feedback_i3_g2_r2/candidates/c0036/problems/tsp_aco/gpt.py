import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0.0) & ~diagonal

    positive = valid & (distance_matrix > 0.0)
    fallback = float(np.median(distance_matrix[positive])) if np.any(positive) else 1.0

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
    if np.any(~rows_with_zero):
        rows = np.flatnonzero(~rows_with_zero)
        row_max = np.max(inverse[rows], axis=1, keepdims=True)
        row_max[row_max <= 0.0] = 1.0
        local[rows] = inverse[rows] / row_max

    if np.any(rows_with_zero):
        local[zero_edges] = 1.0

    top_k = min(n - 1, max(4, int(np.sqrt(n)) + 1))
    tail_weight = 1.0 / np.log2(top_k + 2.0)

    rank_weight = np.full((n, n), tail_weight, dtype=float)
    rank_weight[~valid] = 0.0
    rank_weight[diagonal] = 0.0

    candidate_values = np.where(valid, -local, np.inf)
    top_indices = np.argpartition(candidate_values, kth=top_k - 1, axis=1)[:, :top_k]

    rows = np.arange(n)[:, None]
    selected_valid = valid[rows, top_indices]
    selected_rank = np.arange(1, top_k + 1, dtype=float)[None, :]
    selected_weights = 1.0 / np.log2(selected_rank + 1.0)
    selected_weights = np.broadcast_to(selected_weights, (n, top_k))

    rank_weight[rows, top_indices] = np.where(
        selected_valid,
        selected_weights,
        rank_weight[rows, top_indices],
    )

    score = np.sqrt(
        np.maximum(local, 0.0) * np.maximum(local.T, 0.0)
    )
    score *= np.sqrt(rank_weight * rank_weight.T)
    score = 0.5 * (score + score.T)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
