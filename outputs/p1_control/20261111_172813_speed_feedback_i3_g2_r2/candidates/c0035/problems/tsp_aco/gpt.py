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

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(distance_matrix)
        & (distance_matrix >= 0.0)
        & ~diagonal
    )

    positive = valid & (distance_matrix > 0.0)
    if np.any(positive):
        fallback = float(np.median(distance_matrix[positive]))
    else:
        fallback = 1.0

    safe_distance = np.where(valid, distance_matrix, fallback)
    safe_distance = np.maximum(safe_distance, np.finfo(float).eps)
    safe_distance[diagonal] = np.inf

    inverse_distance = 1.0 / safe_distance
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max[row_max <= 0.0] = 1.0
    local_preference = inverse_distance / row_max
    local_preference[~valid] = 0.0

    order = np.argsort(-local_preference, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values

    rank_weight = 1.0 / np.log2(ranks + 1.0)
    rank_weight[~valid] = 0.0

    endpoint_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )

    rank_compatibility = np.sqrt(
        np.maximum(rank_weight, 0.0)
        * np.maximum(rank_weight.T, 0.0)
    )

    base_score = endpoint_preference * rank_compatibility

    valid_distance = np.where(valid, safe_distance, np.inf)
    sorted_distances = np.sort(valid_distance, axis=1)

    if n >= 2:
        second_nearest = sorted_distances[:, 1]
        nearest = sorted_distances[:, 0]
        second_nearest = np.where(
            np.isfinite(second_nearest),
            second_nearest,
            nearest,
        )
    else:
        second_nearest = np.full(n, fallback, dtype=float)

    second_nearest = np.where(
        np.isfinite(second_nearest) & (second_nearest > 0.0),
        second_nearest,
        fallback,
    )

    alternative_scale = (
        second_nearest[:, None] + second_nearest[None, :]
    )
    edge_distance = safe_distance

    normalized_savings = (
        alternative_scale - edge_distance
    ) / np.maximum(alternative_scale, np.finfo(float).eps)
    normalized_savings = np.clip(normalized_savings, 0.0, 1.0)
    normalized_savings[~valid] = 0.0

    savings_score = endpoint_preference * normalized_savings

    score = base_score * (1.0 + 0.75 * savings_score)
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
