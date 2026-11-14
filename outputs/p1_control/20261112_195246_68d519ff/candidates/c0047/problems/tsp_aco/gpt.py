import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(distances, dtype=float)

    positive_values = distances[valid]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.where(valid, distances, fallback)
    np.fill_diagonal(clean, np.inf)

    eps = np.finfo(float).eps

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / np.maximum(clean[valid], eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    normalized_inverse = np.divide(
        inverse,
        row_max,
        out=np.zeros_like(inverse),
        where=row_max > 0,
    )

    neighbor_count = max(1, min(3, n - 1))
    nearest_mask = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[
                np.argsort(clean[i, candidates], kind="stable")
            ]
            nearest_mask[i, order[:neighbor_count]] = 1.0

    mutual_support = nearest_mask * nearest_mask.T
    local_support = 0.5 * (nearest_mask + nearest_mask.T)

    local_mean = np.zeros(n, dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            count = min(neighbor_count, candidates.size)
            nearest = np.sort(clean[i, candidates], kind="stable")[:count]
            local_mean[i] = float(np.mean(nearest))

    local_mean[local_mean <= 0] = fallback

    local_scale = np.sqrt(
        local_mean[:, None] * local_mean[None, :]
    )
    relative_distance = clean / (local_scale + eps)

    accessibility = np.exp(-relative_distance)
    accessibility[~valid] = 0.0

    score = (
        0.55 * np.sqrt(
            normalized_inverse * normalized_inverse.T
        )
        + 0.25 * accessibility
        + 0.15 * local_support
        + 0.05 * mutual_support
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
