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

    finite_positive = np.isfinite(distances) & (distances > 0.0)
    positive_values = distances[finite_positive]

    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive_values))
    fallback = max(fallback, np.finfo(float).eps)

    work = np.where(finite_positive, distances, fallback)
    work = np.maximum(work, np.finfo(float).eps)
    work = 0.5 * (work + work.T)
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest_values = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest_values, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = work / np.maximum(pair_scale, eps)

    distance_affinity = np.exp(-np.clip(relative_distance, 0.0, 50.0))
    reciprocal_affinity = 1.0 / np.maximum(relative_distance, eps)
    reciprocal_affinity = np.minimum(reciprocal_affinity, 1.0)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_affinity = np.exp(
        -(
            ranks / max(float(n - 1), 1.0)
            + ranks.T / max(float(n - 1), 1.0)
        )
    )

    nearest = np.argmin(work, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    score = (
        0.45 * distance_affinity
        + 0.30 * reciprocal_affinity
        + 0.20 * rank_affinity
        + 0.05 * mutual
    )

    valid = finite_positive & ~np.eye(n, dtype=bool)
    score *= valid
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
