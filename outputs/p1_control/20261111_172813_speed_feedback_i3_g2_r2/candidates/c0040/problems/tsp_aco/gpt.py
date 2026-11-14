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
    zero_distance = valid & (distance_matrix == 0.0)
    positive_distance = valid & (distance_matrix > 0.0)

    positive_values = distance_matrix[positive_distance]
    fallback = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    fallback = max(fallback, np.finfo(float).eps)

    positive_inverse = np.zeros((n, n), dtype=float)
    positive_inverse[positive_distance] = (
        1.0 / distance_matrix[positive_distance]
    )

    row_max = np.max(positive_inverse, axis=1, keepdims=True)
    row_max[row_max <= 0.0] = 1.0

    local_preference = positive_inverse / row_max
    local_preference[zero_distance] = 1.0
    local_preference[~valid] = 0.0

    ranks = np.zeros((n, n), dtype=float)
    for row in range(n):
        zero_columns = np.flatnonzero(zero_distance[row])
        positive_columns = np.flatnonzero(positive_distance[row])

        if zero_columns.size:
            ranks[row, zero_columns] = 1.0

        if positive_columns.size:
            order = positive_columns[
                np.argsort(
                    -local_preference[row, positive_columns],
                    kind="stable",
                )
            ]
            start_rank = zero_columns.size + 1
            ranks[row, order] = np.arange(
                start_rank,
                start_rank + order.size,
                dtype=float,
            )

    rank_weight = np.zeros((n, n), dtype=float)
    rank_weight[valid] = 1.0 / np.log2(ranks[valid] + 1.0)

    endpoint_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )
    rank_compatibility = np.sqrt(
        np.maximum(rank_weight, 0.0)
        * np.maximum(rank_weight.T, 0.0)
    )
    base_score = endpoint_preference * rank_compatibility

    valid_distance = np.where(valid, distance_matrix, np.inf)
    sorted_distances = np.sort(valid_distance, axis=1)
    valid_counts = np.sum(valid, axis=1)
    has_second = valid_counts >= 2

    second_nearest = np.full(n, np.nan, dtype=float)
    second_nearest[has_second] = sorted_distances[has_second, 1]

    finite_second = np.isfinite(second_nearest)
    savings_eligible = (
        valid
        & finite_second[:, None]
        & finite_second[None, :]
    )

    alternative_scale = (
        second_nearest[:, None] + second_nearest[None, :]
    )
    denominator = np.maximum(
        alternative_scale,
        np.finfo(float).eps,
    )

    normalized_savings = (
        alternative_scale - distance_matrix
    ) / denominator
    normalized_savings = np.clip(normalized_savings, 0.0, 1.0)
    normalized_savings[~savings_eligible] = 0.0

    savings_score = endpoint_preference * normalized_savings
    capped_boost = np.minimum(0.5, 0.75 * savings_score)

    score = base_score * (1.0 + capped_boost)
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
