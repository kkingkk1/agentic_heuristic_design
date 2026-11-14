import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directional edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances > 0) & off_diagonal

    positive_values = distances[valid]
    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive_values))
    safe_distances = np.where(valid, distances, fallback)
    safe_distances = np.where(off_diagonal, safe_distances, np.inf)

    row_min = np.min(safe_distances, axis=1)
    row_min = np.maximum(row_min, np.finfo(float).eps)

    relative_length = safe_distances / row_min[:, None]
    base_score = 1.0 / np.maximum(relative_length, np.finfo(float).eps)
    base_score[~valid] = 0.0

    row_max = np.max(base_score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    base_score /= row_max

    ranked_distances = np.sort(
        np.where(valid, safe_distances, np.inf),
        axis=1,
    )
    best = ranked_distances[:, 0]
    second = ranked_distances[:, 1] if n > 1 else best
    second = np.where(np.isfinite(second), second, best)

    regret = np.maximum(second / np.maximum(best, np.finfo(float).eps) - 1.0, 0.0)
    regret = np.minimum(regret, 3.0)

    near_best = np.exp(-1.5 * np.maximum(relative_length - 1.0, 0.0))
    endpoint_factor = 1.0 + 0.35 * regret[:, None] * near_best

    score = base_score * endpoint_factor
    score[~valid] = 0.0

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    score /= row_max

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
