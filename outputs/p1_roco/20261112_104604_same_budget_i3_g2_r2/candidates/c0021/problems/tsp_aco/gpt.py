import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    inverse_distance = np.zeros_like(distances, dtype=float)
    zero_edges = valid & (distances == 0.0)
    positive_edges = valid & (distances > 0.0)

    inverse_distance[zero_edges] = 1.0
    inverse_distance[positive_edges] = 1.0 / (
        distances[positive_edges] + np.finfo(float).eps
    )

    inverse_max = np.max(inverse_distance)
    if inverse_max > 0.0:
        inverse_distance /= inverse_max

    neighborhood = np.zeros_like(distances, dtype=float)

    for i in range(n):
        row_valid = valid[i]
        row_positive = row_valid & (distances[i] > 0.0)
        row_scales = distances[i, row_positive]

        local_scale = (
            float(np.median(row_scales))
            if row_scales.size
            else global_scale
        )
        local_scale = max(local_scale, np.finfo(float).eps)

        neighborhood[i, row_valid] = np.exp(
            -distances[i, row_valid] / local_scale
        )

    neighborhood_similarity = np.sqrt(
        np.maximum(neighborhood, 0.0)
        * np.maximum(neighborhood.T, 0.0)
    )

    scale_bonus = np.zeros_like(distances, dtype=float)
    scale_bonus[valid] = np.exp(
        -distances[valid] / global_scale
    )

    savings = np.zeros_like(distances, dtype=float)

    if n > 2:
        for i in range(n):
            for j in range(n):
                if not valid[i, j] or i == 0 or j == 0:
                    continue

                forward_valid = (
                    valid[i, 0] and valid[0, j]
                )
                reverse_valid = (
                    valid[j, 0] and valid[0, i]
                )

                values = []

                if forward_valid:
                    values.append(
                        distances[i, 0]
                        + distances[0, j]
                        - distances[i, j]
                    )

                if reverse_valid:
                    values.append(
                        distances[j, 0]
                        + distances[0, i]
                        - distances[j, i]
                    )

                if values:
                    savings[i, j] = max(0.0, float(np.mean(values)))

    positive_savings = savings[valid & (savings > 0.0)]
    if positive_savings.size:
        savings_scale = float(np.median(positive_savings))
        savings_scale = max(savings_scale, np.finfo(float).eps)
        savings = np.tanh(savings / savings_scale)
    else:
        savings.fill(0.0)

    scores = (
        0.60 * inverse_distance
        + 0.25 * neighborhood_similarity
        + 0.10 * scale_bonus
        + 0.05 * savings
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0:
        scores /= maximum

    return scores
