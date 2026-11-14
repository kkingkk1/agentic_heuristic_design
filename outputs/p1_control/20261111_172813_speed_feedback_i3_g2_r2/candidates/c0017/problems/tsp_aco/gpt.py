import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    raw = distance_matrix.copy()
    np.fill_diagonal(raw, np.inf)

    finite_edges = np.isfinite(raw) & np.isfinite(raw.T)
    np.fill_diagonal(finite_edges, False)

    positive_values = raw[np.isfinite(raw) & (raw >= 0)]
    fallback = float(np.median(positive_values)) if positive_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    distances = np.where(np.isfinite(raw), raw, fallback)
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    valid_edges = finite_edges & np.isfinite(distances)
    row_has_neighbor = np.any(valid_edges, axis=1)
    neighbor_counts = np.sum(valid_edges, axis=1)
    safe_distances = np.where(valid_edges, distances, np.inf)

    order = np.argsort(safe_distances, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values[None, :]

    reciprocal_rank = np.zeros((n, n), dtype=float)
    reciprocal_rank[valid_edges] = 1.0 / ranks[valid_edges]

    rank_score = reciprocal_rank + reciprocal_rank.T
    rank_max = np.max(rank_score)
    if rank_max > 0:
        rank_score /= rank_max

    sorted_distances = np.take_along_axis(safe_distances, order, axis=1)
    row_best = np.min(safe_distances, axis=1)
    row_second = np.full(n, fallback, dtype=float)

    if n > 1:
        row_second = np.where(
            neighbor_counts >= 2,
            sorted_distances[:, 1],
            sorted_distances[:, 0],
        )

    row_best = np.where(row_has_neighbor, row_best, fallback)
    row_second = np.where(row_has_neighbor, row_second, fallback)

    eps = np.finfo(float).eps

    inverse_distance = np.zeros((n, n), dtype=float)
    inverse_distance[valid_edges] = 1.0 / np.maximum(
        distances[valid_edges],
        eps,
    )

    endpoint_scale = np.max(inverse_distance, axis=1, keepdims=True)
    endpoint_scale[endpoint_scale <= 0] = 1.0
    endpoint_preference = inverse_distance / endpoint_scale

    inverse_score = np.sqrt(
        np.maximum(endpoint_preference, 0.0)
        * np.maximum(endpoint_preference.T, 0.0)
    )
    inverse_score[~valid_edges] = 0.0

    nearest = np.argmin(safe_distances, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        if row_has_neighbor[i]:
            j = nearest[i]
            if row_has_neighbor[j] and nearest[j] == i:
                mutual_nearest[i, j] = 1.0
                mutual_nearest[j, i] = 1.0

    mutual_nearest *= valid_edges

    savings = (
        row_second[:, None]
        + row_second[None, :]
        - 2.0 * distances
    )
    savings = np.maximum(savings, 0.0)
    savings[~valid_edges] = 0.0

    positive_savings = savings[savings > 0]
    if positive_savings.size:
        savings_scale = max(
            float(np.percentile(positive_savings, 90)),
            eps,
        )
        savings_score = np.minimum(savings / savings_scale, 1.0)
    else:
        savings_score = np.zeros_like(savings)

    score = (
        0.70 * rank_score
        + 0.15 * inverse_score
        + 0.10 * mutual_nearest
        + 0.05 * savings_score
    )

    score = 0.5 * (score + score.T)
    score[~valid_edges] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.reshape(distance_matrix.shape)
