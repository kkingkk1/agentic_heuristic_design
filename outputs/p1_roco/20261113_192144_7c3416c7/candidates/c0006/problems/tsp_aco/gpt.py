import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability indicators for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distances.copy()
    diagonal = np.eye(n, dtype=bool)

    finite_values = d[np.isfinite(d) & ~diagonal]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(finite_values)) + 1.0
    d[~np.isfinite(d)] = fallback
    d[diagonal] = np.inf
    d = np.maximum(d, 0.0)

    row_medians = np.median(d, axis=1)
    row_medians[~np.isfinite(row_medians)] = fallback
    row_medians = np.maximum(row_medians, np.finfo(float).eps)

    scaled = d / row_medians[:, None]
    base = 1.0 / (1.0 + scaled)

    nearest = np.argmin(scaled, axis=1)
    mutual_bonus = np.ones((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i and i != j:
            mutual_bonus[i, j] = 2.0

    score = base * mutual_bonus
    score[diagonal] = 0.0

    row_sums = score.sum(axis=1, keepdims=True)
    result = np.divide(
        score,
        row_sums,
        out=np.zeros_like(score),
        where=row_sums > 0,
    )

    return result
