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

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distance_matrix)
    valid = finite & finite.T & ~diagonal

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

    row_scale = np.max(inverse, axis=1, keepdims=True)
    row_scale[row_scale <= 0] = 1.0
    preference = inverse / row_scale
    preference[~valid] = 0.0

    k = min(n - 1, max(2, int(np.ceil(np.sqrt(n)))))

    masked_preference = np.where(valid, preference, -np.inf)
    selected = np.argpartition(
        -masked_preference,
        kth=k - 1,
        axis=1,
    )[:, :k]

    selected_values = np.take_along_axis(
        preference,
        selected,
        axis=1,
    )
    selected_order = np.argsort(
        -selected_values,
        axis=1,
        kind="stable",
    )
    selected = np.take_along_axis(
        selected,
        selected_order,
        axis=1,
    )

    selected_valid = np.take_along_axis(valid, selected, axis=1)
    rank_weights = 1.0 / np.arange(1, k + 1, dtype=float)

    sparse_rank = np.zeros((n, n), dtype=float)
    rows = np.broadcast_to(np.arange(n)[:, None], selected.shape)
    sparse_rank[rows, selected] = (
        selected_valid * rank_weights[None, :]
    )

    sparse_rank = sparse_rank + sparse_rank.T
    sparse_rank_max = np.max(sparse_rank)
    if sparse_rank_max > 0:
        sparse_rank /= sparse_rank_max

    nearest = np.argmin(safe_distances, axis=1)
    has_neighbor = np.any(valid, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    source = np.arange(n)
    nearest_valid = has_neighbor & has_neighbor[nearest]
    mutual = nearest_valid & (nearest[nearest] == source)

    mutual_rows = source[mutual]
    mutual_cols = nearest[mutual]
    mutual_nearest[mutual_rows, mutual_cols] = 1.0
    mutual_nearest[mutual_cols, mutual_rows] = 1.0
    mutual_nearest[~valid] = 0.0

    order = np.argsort(safe_distances, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(
        safe_distances,
        order,
        axis=1,
    )

    row_best = sorted_distances[:, 0]
    row_second = np.where(
        np.sum(valid, axis=1) >= 2,
        sorted_distances[:, 1],
        row_best,
    )

    row_best = np.where(np.isfinite(row_best), row_best, fallback)
    row_second = np.where(np.isfinite(row_second), row_second, fallback)

    savings = (
        row_second[:, None]
        + row_second[None, :]
        - 2.0 * distances
    )
    savings = np.maximum(savings, 0.0)
    savings[~valid] = 0.0

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
        0.48 * preference
        + 0.30 * sparse_rank
        + 0.12 * mutual_nearest
        + 0.10 * savings_score
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.reshape(distance_matrix.shape)
