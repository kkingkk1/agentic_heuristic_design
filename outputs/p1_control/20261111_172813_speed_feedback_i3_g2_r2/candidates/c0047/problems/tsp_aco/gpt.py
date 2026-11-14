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
    original = distance_matrix.copy()

    valid = np.isfinite(original) & (original >= 0.0) & ~diagonal
    positive = valid & (original > 0.0)

    if np.any(positive):
        fallback = float(np.median(original[positive]))
    else:
        fallback = 1.0

    distances = original.copy()
    distances[~valid] = fallback
    distances = np.maximum(distances, np.finfo(float).eps)
    distances[diagonal] = np.inf

    inverse_distance = 1.0 / distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max[row_max <= 0.0] = 1.0
    local_preference = inverse_distance / row_max

    order = np.argsort(-local_preference, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks.fill(n + 1.0)
    rank_values = np.arange(1, n + 1, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values
    ranks[diagonal] = n + 1.0

    rank_weight = 1.0 / np.log2(ranks + 1.0)
    rank_weight[diagonal] = 0.0

    reciprocal_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )
    reciprocal_rank = np.sqrt(
        np.maximum(rank_weight, 0.0)
        * np.maximum(rank_weight.T, 0.0)
    )

    compatibility = reciprocal_preference * reciprocal_rank

    candidate_width = max(2, min(n - 1, int(np.ceil(np.sqrt(n)))))
    neighborhood = np.zeros((n, n), dtype=float)

    for i in range(n):
        nearest = order[i, :candidate_width]
        neighborhood[i, nearest] = 1.0

    common_neighbors = neighborhood @ neighborhood.T
    common_neighbors /= float(candidate_width)
    common_neighbors[diagonal] = 0.0

    compatibility = (
        0.65 * compatibility
        + 0.35 * np.sqrt(
            np.maximum(common_neighbors, 0.0)
            * np.maximum(common_neighbors.T, 0.0)
        )
    )

    savings = np.zeros((n, n), dtype=float)

    if n > 2:
        for i in range(n):
            for j in range(i + 1, n):
                intermediary = np.arange(n)
                intermediary = intermediary[
                    (intermediary != i) & (intermediary != j)
                ]

                values_ij = (
                    distances[i, intermediary]
                    + distances[intermediary, j]
                    - distances[i, j]
                )
                values_ji = (
                    distances[j, intermediary]
                    + distances[intermediary, i]
                    - distances[j, i]
                )

                values = np.concatenate((values_ij, values_ji))
                values = values[np.isfinite(values)]
                values = np.maximum(values, 0.0)

                if values.size:
                    top_count = max(1, int(np.ceil(values.size * 0.25)))
                    top_values = np.partition(
                        values, values.size - top_count
                    )[-top_count:]
                    value = float(np.mean(top_values))
                    savings[i, j] = value
                    savings[j, i] = value

    if np.any(savings > 0.0):
        positive_savings = savings[savings > 0.0]
        savings_scale = float(np.percentile(positive_savings, 90))
        if savings_scale <= 0.0:
            savings_scale = float(np.max(positive_savings))
        savings = np.clip(savings / max(savings_scale, np.finfo(float).eps), 0.0, 1.0)

    distance_signal = inverse_distance.copy()
    distance_max = np.max(distance_signal)
    if distance_max > 0.0:
        distance_signal /= distance_max

    compatibility_max = np.max(compatibility)
    if compatibility_max > 0.0:
        compatibility /= compatibility_max

    score = (
        0.30 * np.sqrt(np.maximum(distance_signal, 0.0))
        + 0.38 * np.sqrt(np.maximum(savings, 0.0))
        + 0.32 * np.sqrt(np.maximum(compatibility, 0.0))
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
