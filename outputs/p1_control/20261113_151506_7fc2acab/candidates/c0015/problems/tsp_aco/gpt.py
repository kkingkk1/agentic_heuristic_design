import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate edge-priority indicators for stochastic TSP solution sampling.

    Larger values indicate edges that are locally attractive from both endpoints
    while remaining competitive under row and column normalization.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros_like(distances, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances)
    valid &= distances >= 0
    valid &= ~np.eye(n, dtype=bool)

    safe = np.where(valid, distances, np.nan)
    positive_values = safe[np.isfinite(safe)]

    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(positive_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    temperature = max(scale * 0.35, np.finfo(float).eps)

    centered = np.where(valid, distances - scale, np.inf)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-centered[valid] / temperature)

    row_sum = affinity.sum(axis=1, keepdims=True)
    col_sum = affinity.sum(axis=0, keepdims=True)

    row_norm = affinity / np.maximum(row_sum, np.finfo(float).eps)
    col_norm = affinity / np.maximum(col_sum, np.finfo(float).eps)

    mutual = np.sqrt(row_norm * col_norm)

    row_best = np.max(np.where(valid, row_norm, 0.0), axis=1, keepdims=True)
    col_best = np.max(np.where(valid, col_norm, 0.0), axis=0, keepdims=True)

    relative_strength = np.sqrt(
        row_norm / np.maximum(row_best, np.finfo(float).eps)
        * col_norm / np.maximum(col_best, np.finfo(float).eps)
    )

    distance_factor = 1.0 / (1.0 + np.maximum(distances, 0.0) / scale)
    distance_factor[~valid] = 0.0

    score = (
        0.45 * mutual
        + 0.35 * relative_strength
        + 0.20 * distance_factor
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
