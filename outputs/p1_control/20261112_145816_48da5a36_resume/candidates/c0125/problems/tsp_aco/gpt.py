import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        valid = distances[finite]
        nonnegative = valid[valid >= 0]
        replacement = float(np.max(nonnegative)) if nonnegative.size else 1.0
        distances = np.where(finite, distances, replacement)

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    finite_offdiag = np.isfinite(distances)
    positive = distances[finite_offdiag & (distances > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps
    scale = max(scale, eps)

    zero_edges = distances == 0.0

    safe_distances = np.where(np.isfinite(distances), distances, scale)
    sorted_distances = np.sort(safe_distances, axis=1)

    local_scale = sorted_distances[:, 0]
    valid_local = np.isfinite(local_scale) & (local_scale > 0)
    local_scale = np.where(valid_local, local_scale, scale)
    local_scale = np.maximum(local_scale, eps)

    normalized_distance = distances / np.sqrt(
        np.outer(local_scale, local_scale)
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 40.0)

    proximity = np.exp(-0.85 * normalized_distance)
    proximity[~np.isfinite(proximity)] = 0.0

    inverse_distance = 1.0 / np.maximum(distances, eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    inverse_distance /= max(float(np.max(inverse_distance)), eps)

    order = np.argsort(safe_distances, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n, dtype=float)
    ranks = np.maximum(ranks, 0.0)

    rank_score = 1.0 / np.power(1.0 + ranks, 0.85)
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True), eps
    )

    mutuality = np.sqrt(rank_score * rank_score.T)
    bilateral_proximity = np.sqrt(proximity * proximity.T)

    score = (
        0.46 * inverse_distance
        + 0.31 * bilateral_proximity
        + 0.17 * mutuality
        + 0.06 * np.sqrt(rank_score * rank_score.T)
    )

    score = 0.5 * (score + score.T)
    score[zero_edges] = np.maximum(score[zero_edges], 1.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
