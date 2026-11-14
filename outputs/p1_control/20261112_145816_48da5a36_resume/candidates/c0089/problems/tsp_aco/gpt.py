import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        valid = distances[finite]
        nonnegative = valid[valid >= 0]
        replacement = float(np.max(nonnegative)) if nonnegative.size else 1.0
        distances = np.where(finite, distances, replacement)

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    diagonal = np.diag_indices(n)
    distances[diagonal] = np.inf

    positive = distances[np.isfinite(distances) & (distances > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    epsilon = np.finfo(float).eps

    nearest = np.min(distances, axis=1)
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        fallback,
    )
    nearest = np.maximum(nearest, epsilon)

    local_scale = np.sqrt(np.outer(nearest, nearest))
    local_score = 1.0 / (1.0 + distances / local_scale)
    local_score[~np.isfinite(local_score)] = 0.0
    local_score[diagonal] = 0.0

    inverse_distance = 1.0 / np.maximum(distances, epsilon)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    maximum = float(np.max(inverse_distance))
    if maximum > 0:
        inverse_distance /= maximum

    nearest_indices = np.argmin(distances, axis=1)
    mutual_bonus = np.zeros((n, n), dtype=float)
    rows = np.arange(n)
    mutual_bonus[rows, nearest_indices] = 1.0
    mutual_bonus[nearest_indices, rows] = 1.0
    mutual_bonus[diagonal] = 0.0

    score = (
        0.55 * local_score
        + 0.30 * inverse_distance
        + 0.15 * mutual_bonus
    )
    score = 0.5 * (score + score.T)
    score[diagonal] = 0.0

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
