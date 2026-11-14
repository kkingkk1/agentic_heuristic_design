import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a two-dimensional array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros_like(distance_matrix, dtype=float)

    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
    )
    np.fill_diagonal(valid, False)

    positive_values = distance_matrix[valid & (distance_matrix > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
        epsilon = max(float(np.min(positive_values)) * 1e-6, 1e-12)
    else:
        global_scale = 1.0
        epsilon = 1e-12

    row_scale = np.full(n, global_scale, dtype=float)
    col_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row_values = distance_matrix[i, valid[i]]
        row_positive = row_values[row_values > 0.0]
        if row_positive.size:
            row_scale[i] = max(float(np.median(row_positive)), epsilon)

    for j in range(n):
        col_values = distance_matrix[valid[:, j], j]
        col_positive = col_values[col_values > 0.0]
        if col_positive.size:
            col_scale[j] = max(float(np.median(col_positive)), epsilon)

    safe_distance = np.where(valid, np.maximum(distance_matrix, 0.0), 0.0)

    outgoing_affinity = np.exp(
        -safe_distance / row_scale[:, None]
    )
    incoming_affinity = np.exp(
        -safe_distance / col_scale[None, :]
    )

    rank_score = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            ordered = candidates[np.argsort(distance_matrix[i, candidates], kind="stable")]
            rank_score[i, ordered] = 1.0 / (1.0 + np.arange(ordered.size))

    score = (
        0.50 * outgoing_affinity
        + 0.30 * incoming_affinity
        + 0.20 * rank_score
    )

    score[~valid] = 0.0

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score
