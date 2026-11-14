import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        valid = distances[finite & (distances >= 0.0)]
        replacement = float(np.max(valid)) if valid.size else 1.0
        distances = np.where(finite, distances, replacement)

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    eps = np.finfo(float).eps
    np.fill_diagonal(distances, np.inf)

    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, eps)

    nearest = np.min(distances, axis=1)
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        fallback,
    )
    nearest = np.maximum(nearest, eps)

    scale = np.sqrt(nearest[:, None] * nearest[None, :])
    scale = np.maximum(scale, eps)

    normalized_distance = distances / scale
    normalized_distance = np.maximum(normalized_distance, 0.0)

    adaptive_kernel = np.exp(-normalized_distance)
    adaptive_kernel[~np.isfinite(adaptive_kernel)] = 0.0
    np.fill_diagonal(adaptive_kernel, 0.0)

    inverse_distance = 1.0 / np.maximum(distances, eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    inverse_distance = inverse_distance / np.maximum(row_max, eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    positions = np.broadcast_to(
        np.arange(n, dtype=float),
        (n, n),
    )
    np.put_along_axis(ranks, order, positions, axis=1)

    rank_weight = 1.0 / (1.0 + ranks)
    rank_weight /= np.maximum(
        np.max(rank_weight, axis=1, keepdims=True),
        eps,
    )
    mutual_rank = np.sqrt(rank_weight * rank_weight.T)
    mutual_rank[~np.isfinite(mutual_rank)] = 0.0
    np.fill_diagonal(mutual_rank, 0.0)

    scale_ratio = np.minimum(nearest[:, None], nearest[None, :]) / np.maximum(
        nearest[:, None],
        nearest[None, :],
    )
    scale_ratio = np.clip(scale_ratio, 0.0, 1.0)

    score = (
        0.50 * inverse_distance
        + 0.32 * adaptive_kernel
        + 0.18 * mutual_rank
    )
    score *= 0.90 + 0.10 * scale_ratio
    score = 0.5 * (score + score.T)

    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
