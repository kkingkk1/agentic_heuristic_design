import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    distances = np.maximum(distance_matrix, 0.0).copy()
    np.fill_diagonal(distances, np.inf)

    finite = np.isfinite(distances)
    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[finite]
    positive_values = positive_values[positive_values > 0.0]

    if positive_values.size:
        global_scale = np.percentile(positive_values, 40.0)
    else:
        global_scale = 1.0

    global_scale = max(float(global_scale), np.finfo(float).eps)

    nearest = np.min(distances, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]

    if finite_nearest.size:
        local_scale = np.where(
            np.isfinite(nearest),
            nearest,
            np.median(finite_nearest),
        )
        adaptive_scale = 0.5 * local_scale + 0.5 * global_scale
    else:
        adaptive_scale = np.full(n, global_scale, dtype=float)

    adaptive_scale = np.maximum(adaptive_scale, np.finfo(float).eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized_distance = distances / np.sqrt(
            np.outer(adaptive_scale, adaptive_scale)
        )

        desirability = 1.0 / (1.0 + normalized_distance**3.2)

        row_order = np.argsort(distances, axis=1)
        ranks = np.empty_like(row_order, dtype=float)
        ranks[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
        rank_weight = 1.0 / (1.0 + ranks) ** 0.8

        local_count = max(1, min(5, n - 1))
        local_neighbors = row_order[:, :local_count]
        neighborhood_mask = np.zeros((n, n), dtype=float)
        neighborhood_mask[
            np.arange(n)[:, None],
            local_neighbors,
        ] = 1.0

        reverse_support = neighborhood_mask.T
        neighborhood_support = 0.5 * (
            neighborhood_mask + reverse_support
        )

        row_nearest = np.argmin(distances, axis=1)
        mutual = np.zeros((n, n), dtype=float)

        valid_nearest = np.isfinite(nearest)
        for i in range(n):
            j = row_nearest[i]
            if valid_nearest[i] and j != i and row_nearest[j] == i:
                mutual[i, j] = 1.0

        local_ratio = np.sqrt(np.outer(adaptive_scale, adaptive_scale))
        local_ratio /= distances + 0.5 * global_scale
        local_ratio = np.clip(local_ratio, 0.0, 1.0)

        scores = desirability**0.65
        scores *= (0.72 + 0.28 * rank_weight)
        scores *= 0.78 + 0.22 * neighborhood_support
        scores *= 0.78 + 0.22 * local_ratio
        scores *= 1.0 + 0.35 * mutual

    scores[~finite] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
