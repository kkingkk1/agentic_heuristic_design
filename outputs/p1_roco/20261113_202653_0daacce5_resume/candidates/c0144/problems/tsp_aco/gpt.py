import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_distances = distances[valid]
    scale = float(np.median(positive_distances))
    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid] = scale / (distances[valid] + eps)

    valid_inverse = inverse_distance[valid]
    lower = float(np.quantile(valid_inverse, 0.02))
    upper = float(np.quantile(valid_inverse, 0.98))

    if upper > lower:
        inverse_distance = np.clip(inverse_distance, lower, upper)
        inverse_distance[valid] = (
            inverse_distance[valid] - lower
        ) / (upper - lower)
    else:
        inverse_distance[valid] = 1.0

    row_competitiveness = np.zeros((n, n), dtype=float)
    col_competitiveness = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_mask = valid[i]
        row_values = inverse_distance[i, row_mask]
        if row_values.size:
            q25, q75 = np.quantile(row_values, [0.25, 0.75])
            spread = max(float(q75 - q25), eps)
            row_competitiveness[i, row_mask] = np.clip(
                (row_values - q25) / spread,
                0.0,
                1.0,
            )

    for j in range(n):
        col_mask = valid[:, j]
        col_values = inverse_distance[col_mask, j]
        if col_values.size:
            q25, q75 = np.quantile(col_values, [0.25, 0.75])
            spread = max(float(q75 - q25), eps)
            col_competitiveness[col_mask, j] = np.clip(
                (col_values - q25) / spread,
                0.0,
                1.0,
            )

    local_signal = (
        0.5
        + 0.25 * row_competitiveness
        + 0.25 * col_competitiveness
    )

    priors = inverse_distance * local_signal

    exploration_floor = 0.03
    priors[valid] = (
        (1.0 - exploration_floor) * priors[valid]
        + exploration_floor
    )

    priors[~valid] = 0.0
    np.fill_diagonal(priors, 0.0)

    maximum = float(np.max(priors))
    if maximum > 0.0:
        priors /= maximum

    return priors
