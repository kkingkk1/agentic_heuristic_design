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
    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0.0) & ~diagonal
    positive = valid & (distance_matrix > 0.0)

    if np.any(positive):
        fallback = float(np.median(distance_matrix[positive]))
    else:
        fallback = 1.0

    distances = np.array(distance_matrix, dtype=float, copy=True)
    distances[~valid] = fallback
    distances = np.maximum(distances, np.finfo(float).eps)
    distances[diagonal] = np.inf

    inverse_distance = 1.0 / distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max[row_max <= 0.0] = 1.0
    local_preference = inverse_distance / row_max

    incoming_max = np.max(inverse_distance, axis=0, keepdims=True)
    incoming_max[incoming_max <= 0.0] = 1.0
    incoming_preference = inverse_distance / incoming_max

    reciprocal = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(incoming_preference, 0.0)
    )

    order = np.argsort(-local_preference, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values
    rank_weight = 1.0 / np.log2(ranks + 1.0)
    rank_weight[diagonal] = 0.0

    outgoing_best = np.min(distances, axis=1)
    incoming_best = np.min(distances, axis=0)
    savings = (
        outgoing_best[:, None]
        + incoming_best[None, :]
        - distances
    )
    savings = np.maximum(savings, 0.0)
    savings[~valid] = 0.0

    positive_savings = savings[savings > 0.0]
    if positive_savings.size:
        savings_scale = float(np.percentile(positive_savings, 90))
        if savings_scale <= 0.0:
            savings_scale = float(np.max(positive_savings))
        savings = np.clip(savings / savings_scale, 0.0, 1.0)

    score = (
        0.45 * savings
        + 0.40 * reciprocal
        + 0.15 * reciprocal * rank_weight
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    candidate_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(n))) + 1)
    )
    candidate_mask = np.zeros((n, n), dtype=bool)

    candidate_order = np.argsort(-score, axis=1, kind="stable")
    candidate_mask[
        np.arange(n)[:, None],
        candidate_order[:, :candidate_count]
    ] = True

    nearest_order = np.argsort(-local_preference, axis=1, kind="stable")
    candidate_mask[
        np.arange(n),
        nearest_order[:, 0]
    ] = True

    mutual_nearest = np.zeros((n, n), dtype=bool)
    nearest = nearest_order[:, 0]
    mutual_nearest[np.arange(n), nearest] = True
    mutual_nearest &= mutual_nearest.T
    candidate_mask |= mutual_nearest

    score[~candidate_mask] = 0.0
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
