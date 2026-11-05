import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability scores for a fixed ant-colony TSP solver."""
    distances = np.asarray(distance_matrix, dtype=np.float64)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=np.float64)

    if n <= 1:
        return scores

    off_diagonal = ~np.eye(n, dtype=bool)
    positive_distances = distances[off_diagonal]
    positive_distances = positive_distances[positive_distances > 0]

    if positive_distances.size == 0:
        return scores

    global_scale = float(np.median(positive_distances))
    epsilon = np.finfo(np.float64).eps * max(global_scale, 1.0)

    row_scales = np.empty(n, dtype=np.float64)
    for i in range(n):
        row_values = distances[i, np.arange(n) != i]
        row_values = row_values[row_values > 0]
        row_scales[i] = (
            float(np.median(row_values)) if row_values.size else global_scale
        )

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            distance = max(float(distances[i, j]), 0.0)
            local_scale = np.sqrt(row_scales[i] * row_scales[j])
            local_scale = max(local_scale, epsilon)

            inverse_distance = 1.0 / (1.0 + distance / local_scale)

            alternatives_i = distances[i].copy()
            alternatives_i[i] = np.inf
            alternatives_i[j] = np.inf

            alternatives_j = distances[j].copy()
            alternatives_j[j] = np.inf
            alternatives_j[i] = np.inf

            if n > 2:
                rank_i = np.count_nonzero(alternatives_i <= distance) / (n - 2)
                rank_j = np.count_nonzero(alternatives_j <= distance) / (n - 2)
            else:
                rank_i = 0.0
                rank_j = 0.0

            local_competitiveness = (1.0 - rank_i) * (1.0 - rank_j)

            if n > 2:
                detour_candidates = (
                    distances[i, :] + distances[:, j]
                ).astype(np.float64)
                detour_candidates[i] = np.inf
                detour_candidates[j] = np.inf
                best_two_hop = float(np.min(detour_candidates))
                detour_advantage = best_two_hop / (
                    best_two_hop + distance + epsilon
                )
            else:
                detour_advantage = 0.5

            scores[i, j] = (
                0.50 * inverse_distance
                + 0.30 * local_competitiveness
                + 0.20 * detour_advantage
            )

    return scores
