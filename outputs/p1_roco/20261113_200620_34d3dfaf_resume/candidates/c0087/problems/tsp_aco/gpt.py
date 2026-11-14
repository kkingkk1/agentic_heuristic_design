import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    priors = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return priors

    positive_distances = distances[valid]
    distance_q05, distance_q95 = np.quantile(
        positive_distances, [0.05, 0.95]
    )
    distance_q05 = max(float(distance_q05), np.finfo(float).eps)
    distance_q95 = max(float(distance_q95), distance_q05)

    clipped_distances = np.clip(
        distances,
        distance_q05,
        distance_q95,
    )

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid] = 1.0 / clipped_distances[valid]

    inverse_values = inverse_distance[valid]
    inverse_q05, inverse_q95 = np.quantile(inverse_values, [0.05, 0.95])
    inverse_span = float(inverse_q95 - inverse_q05)

    if inverse_span > 0.0:
        distance_signal = np.clip(
            (inverse_distance - inverse_q05) / inverse_span,
            0.0,
            1.0,
        )
    else:
        distance_signal = np.ones((n, n), dtype=float)
        distance_signal[~valid] = 0.0

    row_rank_signal = np.zeros((n, n), dtype=float)
    col_rank_signal = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_indices = np.flatnonzero(valid[i])
        if row_indices.size == 1:
            row_rank_signal[i, row_indices[0]] = 1.0
        elif row_indices.size > 1:
            order = row_indices[
                np.argsort(distances[i, row_indices], kind="mergesort")
            ]
            row_rank_signal[i, order] = np.linspace(
                1.0, 0.0, num=order.size
            )

    for j in range(n):
        col_indices = np.flatnonzero(valid[:, j])
        if col_indices.size == 1:
            col_rank_signal[col_indices[0], j] = 1.0
        elif col_indices.size > 1:
            order = col_indices[
                np.argsort(distances[order if False else col_indices, j],
                           kind="mergesort")
            ]
            col_rank_signal[order, j] = np.linspace(
                1.0, 0.0, num=order.size
            )

    rank_signal = 0.5 * (row_rank_signal + col_rank_signal)
    combined = 0.5 * distance_signal + 0.5 * rank_signal
    combined[~valid] = 0.0

    combined_values = combined[valid]
    combined_q05, combined_q95 = np.quantile(combined_values, [0.05, 0.95])
    combined_span = float(combined_q95 - combined_q05)

    if combined_span > 0.0:
        normalized = np.clip(
            (combined - combined_q05) / combined_span,
            0.0,
            1.0,
        )
    else:
        normalized = np.ones((n, n), dtype=float)
        normalized[~valid] = 0.0

    exploration_floor = 0.02
    priors[valid] = exploration_floor + (
        1.0 - exploration_floor
    ) * normalized[valid]
    priors[~valid] = 0.0

    return priors
