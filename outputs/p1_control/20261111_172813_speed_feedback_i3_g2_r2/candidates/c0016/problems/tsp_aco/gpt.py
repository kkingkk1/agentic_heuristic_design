import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distance_matrix)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite & finite.T & off_diagonal

    positive = valid & (distance_matrix > 0)
    if np.any(positive):
        fallback = float(np.median(distance_matrix[positive]))
    else:
        fallback = 1.0

    eps = np.finfo(float).eps
    fallback = max(fallback, eps)

    distances = np.where(finite, distance_matrix, fallback)
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    safe_distances = np.where(valid, distances, np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(distances[valid], eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    preference = inverse / row_max
    preference[~valid] = 0.0

    k = min(n - 1, max(2, int(np.ceil(np.sqrt(n)))))

    masked_preference = np.where(valid, preference, -np.inf)
    selected = np.argpartition(
        -masked_preference,
        kth=k - 1,
        axis=1,
    )[:, :k]

    selected_values = np.take_along_axis(preference, selected, axis=1)
    selected_order = np.argsort(
        -selected_values,
        axis=1,
        kind="stable",
    )
    selected = np.take_along_axis(selected, selected_order, axis=1)

    selected_valid = np.take_along_axis(valid, selected, axis=1)
    rank_weights = 1.0 / np.arange(1, k + 1, dtype=float)
    selected_scores = selected_valid * rank_weights[None, :]

    scarcity = np.zeros((n, n), dtype=float)
    rows = np.broadcast_to(np.arange(n)[:, None], selected.shape)
    scarcity[rows, selected] = selected_scores
    scarcity = scarcity + scarcity.T

    scarcity_max = np.max(scarcity)
    if scarcity_max > 0:
        scarcity /= scarcity_max

    candidate_mask = (scarcity > 0) & valid

    nearest = np.argmin(safe_distances, axis=1)
    has_neighbor = np.any(valid, axis=1)

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        if has_neighbor[i]:
            j = nearest[i]
            if has_neighbor[j] and nearest[j] == i:
                mutual_nearest[i, j] = 1.0
                mutual_nearest[j, i] = 1.0

    mutual_nearest *= candidate_mask

    order = np.argsort(safe_distances, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(safe_distances, order, axis=1)

    row_best = sorted_distances[:, 0]
    row_second = np.where(
        np.sum(valid, axis=1) >= 2,
        sorted_distances[:, 1],
        row_best,
    )
    row_best = np.where(has_neighbor, row_best, fallback)
    row_second = np.where(has_neighbor, row_second, fallback)

    savings = row_second[:, None] + row_second[None, :] - 2.0 * distances
    savings = np.maximum(savings, 0.0)
    savings[~candidate_mask] = 0.0

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
        0.58 * scarcity
        + 0.27 * preference
        + 0.10 * mutual_nearest
        + 0.05 * savings_score
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.reshape(distance_matrix.shape)
