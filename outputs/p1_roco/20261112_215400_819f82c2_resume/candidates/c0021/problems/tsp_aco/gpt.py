import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate edge-selection priors for stochastic TSP solution sampling.

    The score combines:
    1. Distance normalized by the local scale of both endpoints.
    2. Reciprocal-neighbor preference from both endpoints.
    3. A mutual-nearest-neighbor bonus.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    finite_positive = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(finite_positive, False)

    safe_distances = np.where(finite_positive, distances, np.inf)

    row_scales = np.full(n, 1.0, dtype=float)
    for i in range(n):
        values = safe_distances[i, finite_positive[i]]
        if values.size:
            row_scales[i] = max(float(np.median(values)), np.finfo(float).eps)

    normalized = distances / np.sqrt(row_scales[:, None] * row_scales[None, :])
    normalized = np.where(finite_positive, normalized, np.inf)

    local_affinity = np.exp(-np.minimum(normalized, 50.0))

    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        valid = finite_positive[i]
        order = np.argsort(safe_distances[i], kind="stable")
        valid_order = order[valid[order]]
        ranks[i, valid_order] = np.arange(valid_order.size, dtype=float)

    reciprocal_rank = 1.0 / (1.0 + ranks)
    reciprocal_support = np.sqrt(reciprocal_rank * reciprocal_rank.T)

    nearest = np.argmin(safe_distances, axis=1)
    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if np.isfinite(safe_distances[i, j]) and nearest[j] == i:
            mutual_bonus[i, j] = 1.0

    scores = (
        0.55 * local_affinity
        + 0.35 * reciprocal_support
        + 0.10 * mutual_bonus
    )

    scores = np.where(finite_positive, scores, 0.0)
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0:
        scores /= maximum

    return scores.astype(float, copy=False)
