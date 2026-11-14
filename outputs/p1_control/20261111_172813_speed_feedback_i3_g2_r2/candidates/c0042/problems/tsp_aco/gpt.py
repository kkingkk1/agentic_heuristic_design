import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

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

    inverse = np.zeros((n, n), dtype=float)
    inverse[positive_distance] = 1.0 / distance_matrix[positive_distance]

    local_preference = np.zeros((n, n), dtype=float)
    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max = np.maximum(row_max, np.finfo(float).eps)
    local_preference[positive_distance] = (
        inverse[positive_distance] / row_max[
            np.broadcast_to(np.arange(n)[:, None], (n, n))
        ][positive_distance]
    )
    local_preference[zero_distance] = 1.0
    local_preference[~valid] = 0.0

    valid_counts = np.sum(valid, axis=1)
    top_k = min(n - 1, max(3, int(np.sqrt(n))))
    top_k = max(1, top_k)

    tail_weight = 1.0 / np.log2(top_k + 2.0)
    weighted = local_preference * tail_weight

    candidate_values = np.where(valid, -local_preference, np.inf)
    top_indices = np.argpartition(
        candidate_values,
        kth=top_k - 1,
        axis=1,
    )[:, :top_k]

    rows = np.arange(n)[:, None]
    selected_local = local_preference[rows, top_indices]

    order = np.argsort(
        -selected_local,
        axis=1,
        kind="stable",
    )
    ranked_indices = np.take_along_axis(top_indices, order, axis=1)
    ranked_local = np.take_along_axis(selected_local, order, axis=1)

    ranks = np.arange(1, top_k + 1, dtype=float)[None, :]
    rank_weight = 1.0 / np.log2(ranks + 1.0)

    confidence = ranked_local * ranked_local
    selected_weight = (
        tail_weight
        + confidence * (rank_weight - tail_weight)
    )
    selected_weighted = ranked_local * selected_weight

    selected_valid = valid[rows, ranked_indices]
    selected_weighted = np.where(
        selected_valid,
        selected_weighted,
        0.0,
    )
    weighted[rows, ranked_indices] = selected_weighted

    endpoint_preference = np.sqrt(
        np.maximum(weighted, 0.0)
        * np.maximum(weighted.T, 0.0)
    )

    rank_support = np.zeros((n, n), dtype=float)
    rank_support[rows, ranked_indices] = rank_weight
    rank_support[~valid] = 0.0

    rank_compatibility = np.sqrt(
        np.maximum(rank_support, 0.0)
        * np.maximum(rank_support.T, 0.0)
    )

    base_score = endpoint_preference * (
        0.5 + 0.5 * rank_compatibility
    )

    valid_distance = np.where(valid, distance_matrix, np.inf)
    sorted_distances = np.sort(valid_distance, axis=1)

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
