import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    positive_finite = np.isfinite(distances) & (distances > 0)
    if not np.any(positive_finite):
        return np.zeros_like(distances, dtype=float)

    fallback = float(np.max(distances[positive_finite]))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.array(distances, dtype=float, copy=True)
    clean[~np.isfinite(clean)] = fallback
    clean[clean <= 0] = fallback
    np.fill_diagonal(clean, np.inf)

    valid = np.isfinite(clean)
    eps = np.finfo(float).eps

    inverse = np.zeros_like(clean)
    inverse[valid] = 1.0 / np.maximum(clean[valid], eps)
    inverse /= np.maximum(np.max(inverse, axis=1, keepdims=True), 1.0)

    nearest_distances = np.min(
        np.where(valid, clean, np.inf),
        axis=1,
        keepdims=True,
    )
    nearest_distances[~np.isfinite(nearest_distances)] = fallback

    closeness = np.zeros_like(clean)
    row_indices = np.broadcast_to(
        np.arange(n)[:, None],
        clean.shape,
    )
    closeness[valid] = np.exp(
        -(
            clean[valid]
            - nearest_distances[row_indices[valid], 0]
        )
        / (nearest_distances[row_indices[valid], 0] + eps)
    )
    closeness /= np.maximum(np.max(closeness, axis=1, keepdims=True), 1.0)

    order = np.argsort(clean, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = np.zeros_like(clean)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])
    rank_score /= np.maximum(np.max(rank_score, axis=1, keepdims=True), 1.0)

    nearest = np.argmin(clean, axis=1)
    mutual = np.zeros_like(clean)
    for i, j in enumerate(nearest):
        if i != j and nearest[j] == i and valid[i, j]:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    local_score = (
        0.50 * inverse
        + 0.28 * closeness
        + 0.08 * rank_score
        + 0.14 * mutual
    )

    local_score[~valid] = 0.0
    np.fill_diagonal(local_score, 0.0)

    score = 0.5 * (local_score + local_score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
