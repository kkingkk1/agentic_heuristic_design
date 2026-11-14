import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_positive = np.isfinite(distances) & (distances > 0)
    positive_values = distances[finite_positive]

    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(positive_values))
    fallback = max(fallback, np.finfo(float).eps)

    safe = np.where(finite_positive, distances, fallback)
    safe = 0.5 * (safe + safe.T)
    np.fill_diagonal(safe, np.inf)

    finite_edges = safe[np.isfinite(safe) & (safe > 0)]
    scale = (
        float(np.median(finite_edges))
        if finite_edges.size
        else fallback
    )
    scale = max(scale, np.finfo(float).eps)

    local_scale = np.min(safe, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    relative_length = safe / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    inverse_distance = np.zeros((n, n), dtype=float)
    finite_mask = np.isfinite(safe) & (safe > 0)
    inverse_distance[finite_mask] = 1.0 / safe[finite_mask]

    inverse_max = np.max(inverse_distance, axis=1, keepdims=True)
    inverse_term = np.divide(
        inverse_distance,
        inverse_max,
        out=np.zeros_like(inverse_distance),
        where=inverse_max > 0,
    )

    ranks = np.argsort(np.argsort(safe, axis=1), axis=1).astype(float)
    rank_term = np.exp(
        -0.8 * (ranks + ranks.T) / max(1.0, np.sqrt(float(n)))
    )

    nearest = np.argmin(safe, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest[i])
        if 0 <= j < n and nearest[j] == i and i != j:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    proximity_term = 1.0 / (1.0 + relative_length)
    long_edge_penalty = np.exp(
        -0.7 * np.maximum(relative_length - 1.0, 0.0)
    )

    local_term = proximity_term * long_edge_penalty
    local_term[~np.isfinite(local_term)] = 0.0

    score = (
        0.45 * inverse_term
        + 0.30 * local_term
        + 0.25 * rank_term
    )
    score *= 1.0 + 0.25 * mutual_nearest
    score *= finite_mask & finite_mask.T

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
