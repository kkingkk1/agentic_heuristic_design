import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    distances = distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)

    finite_positive = np.isfinite(distances) & (distances > 0)
    if not np.any(finite_positive):
        return np.zeros_like(distances, dtype=float)

    fallback = np.max(distances[finite_positive])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    safe_distances = np.where(
        finite_positive,
        distances,
        fallback,
    )

    inverse = 1.0 / (safe_distances + np.finfo(float).eps)
    inverse[~finite_positive] = 0.0

    row_max = np.max(inverse, axis=1, keepdims=True)
    normalized_inverse = np.divide(
        inverse,
        row_max,
        out=np.zeros_like(inverse),
        where=row_max > 0,
    )

    neighbor_count = max(1, min(4, n - 1))
    nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        valid = np.flatnonzero(finite_positive[i])
        if valid.size:
            order = valid[np.argsort(safe_distances[i, valid], kind="stable")]
            nearest[i, order[:neighbor_count]] = 1.0

    mutual_bonus = nearest * nearest.T
    one_sided_bonus = np.maximum(nearest, nearest.T)

    scores = (
        0.72 * normalized_inverse
        + 0.18 * one_sided_bonus
        + 0.10 * mutual_bonus
    )

    scores[~finite_positive] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
