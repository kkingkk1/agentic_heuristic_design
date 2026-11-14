import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)
    symmetric_valid = valid & valid.T

    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distances = 0.5 * (
        np.where(valid, distances, 0.0)
        + np.where(valid.T, distances.T, 0.0)
    )
    symmetric_distances[~symmetric_valid] = np.inf
    np.fill_diagonal(symmetric_distances, np.inf)

    positive = symmetric_distances[
        np.isfinite(symmetric_distances)
        & (symmetric_distances > 0.0)
    ]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    inverse_distance = 1.0 / (symmetric_distances + eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    np.fill_diagonal(inverse_distance, 0.0)

    row_peak = np.max(inverse_distance, axis=1, keepdims=True)
    col_peak = np.max(inverse_distance, axis=0, keepdims=True)

    row_competitiveness = inverse_distance / np.maximum(row_peak, eps)
    col_competitiveness = inverse_distance / np.maximum(col_peak, eps)

    proximity = np.exp(-symmetric_distances / (0.90 * scale))
    proximity[~np.isfinite(proximity)] = 0.0
    np.fill_diagonal(proximity, 0.0)

    row_mass = proximity.sum(axis=1)
    local_preference = proximity / np.maximum(row_mass[:, None], eps)
    mutual_support = np.sqrt(
        np.maximum(local_preference * local_preference.T, 0.0)
    )

    scarcity = 1.0 / np.maximum(row_mass, eps)
    scarcity /= max(float(np.mean(scarcity)), eps)

    endpoint_support = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    endpoint_support = np.clip(endpoint_support, 0.0, 2.5)

    competitiveness = (
        0.45 * row_competitiveness
        + 0.55 * col_competitiveness
    )

    score = inverse_distance
    score *= 0.50 + 0.30 * competitiveness
    score *= 0.72 + 0.28 * endpoint_support
    score *= 0.68 + 0.68 * mutual_support
    score *= 0.78 + 0.22 * proximity

    score[~symmetric_valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
