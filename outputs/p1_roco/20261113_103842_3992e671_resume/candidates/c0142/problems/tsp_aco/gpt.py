import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distance_matrix)
    valid = finite & ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return scores

    positive_values = distance_matrix[valid & (distance_matrix > 0.0)]
    if positive_values.size:
        floor = max(
            np.finfo(float).eps,
            float(np.percentile(positive_values, 5.0)) * 1e-3,
        )
    else:
        floor = np.finfo(float).eps

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = np.maximum(distance_matrix[valid], floor)

    symmetric_valid = valid & valid.T
    symmetric_distance = np.full((n, n), np.inf, dtype=float)
    symmetric_distance[symmetric_valid] = (
        distances[symmetric_valid] + distances.T[symmetric_valid]
    ) * 0.5

    usable = np.isfinite(symmetric_distance)
    if not np.any(usable):
        usable = valid

    edge_values = symmetric_distance[usable]
    scale = float(np.percentile(edge_values, 50.0))
    scale = max(scale, floor)

    distance_score = np.zeros((n, n), dtype=float)
    distance_score[usable] = 1.0 / (
        1.0 + (symmetric_distance[usable] / scale) ** 2.2
    )

    row_min = np.full(n, scale, dtype=float)
    for i in range(n):
        row_values = distances[i]
        finite_row = row_values[np.isfinite(row_values)]
        if finite_row.size:
            row_min[i] = max(float(np.min(finite_row)), floor)

    local_score = np.zeros((n, n), dtype=float)
    local_scale = np.sqrt(np.outer(row_min, row_min))
    local_score[valid] = local_scale[valid] / (
        local_scale[valid] + distances[valid]
    )

    rank_score = np.zeros((n, n), dtype=float)
    for i in range(n):
        finite_indices = np.flatnonzero(np.isfinite(distances[i]))
        if finite_indices.size:
            order = finite_indices[np.argsort(distances[i, finite_indices])]
            rank_score[i, order] = 1.0 - (
                np.arange(order.size, dtype=float) / max(order.size - 1, 1)
            )

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        finite_indices = np.flatnonzero(np.isfinite(distances[i]))
        if finite_indices.size:
            nearest[i] = finite_indices[np.argmin(distances[i, finite_indices])]

    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual[i, j] = 1.0

    directional_rank = 0.5 * (rank_score + rank_score.T)

    combined = (
        0.58 * distance_score
        + 0.18 * local_score
        + 0.16 * directional_rank
        + 0.08 * mutual
    )

    combined[~valid] = 0.0
    np.fill_diagonal(combined, 0.0)

    maximum = float(np.max(combined))
    if maximum > 0.0 and np.isfinite(maximum):
        combined /= maximum

    return combined
