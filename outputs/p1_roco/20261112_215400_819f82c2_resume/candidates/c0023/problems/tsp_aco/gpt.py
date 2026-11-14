import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square two-dimensional array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distance_matrix)
    positive = distance_matrix > 0
    valid = finite & positive

    safe_distance = np.where(valid, distance_matrix, np.nan)

    row_median = np.nanmedian(safe_distance, axis=1)
    global_median = np.nanmedian(safe_distance)

    if not np.isfinite(global_median) or global_median <= 0:
        global_median = 1.0

    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0),
        row_median,
        global_median,
    )

    scaled = np.where(
        valid,
        distance_matrix / row_median[:, None],
        np.inf,
    )

    inverse_distance = np.where(
        valid,
        1.0 / np.maximum(scaled, 1e-12),
        0.0,
    )
    inverse_distance = np.log1p(inverse_distance)

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    inverse_distance = inverse_distance / row_max

    ranks = np.zeros_like(distance_matrix, dtype=float)
    for i in range(n):
        order = np.argsort(np.where(valid[i], distance_matrix[i], np.inf))
        valid_order = order[valid[i, order]]
        count = valid_order.size
        if count > 0:
            ranks[i, valid_order] = np.arange(count, 0, -1, dtype=float) / count

    local_density = np.zeros_like(distance_matrix, dtype=float)
    for j in range(n):
        incoming = distance_matrix[:, j]
        incoming_valid = valid[:, j]
        if np.any(incoming_valid):
            reference = np.median(incoming[incoming_valid])
            local_density[:, j] = np.where(
                incoming_valid,
                1.0 / (1.0 + incoming / max(reference, 1e-12)),
                0.0,
            )

    score = (
        0.55 * inverse_distance
        + 0.30 * ranks
        + 0.15 * local_density
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
