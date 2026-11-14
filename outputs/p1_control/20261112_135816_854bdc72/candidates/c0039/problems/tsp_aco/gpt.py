import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return topology-aware symmetric edge-priority indicators for TSP sampling."""
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
        finite_values = distances[finite]
        nonnegative = finite_values[finite_values >= 0]
        replacement = float(np.max(nonnegative)) if nonnegative.size else 1.0
        distances = np.where(finite, distances, replacement)

    if np.any(distances < 0):
        raise ValueError("distances must be non-negative")

    distances = 0.5 * (distances + distances.T)
    valid = np.isfinite(distances) & (distances >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    working = np.where(valid, distances, np.inf)
    order = np.argsort(working, axis=1, kind="stable")

    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    percentile = rank / max(n - 1, 1)
    percentile[~valid] = 1.0

    affinity = np.exp(-3.5 * percentile)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_mass = np.sum(affinity, axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, np.finfo(float).eps)
    normalized_affinity = affinity / row_mass

    mutual_preference = np.sqrt(
        normalized_affinity * normalized_affinity.T
    )

    neighborhood_overlap = normalized_affinity @ normalized_affinity.T
    neighborhood_overlap = np.clip(neighborhood_overlap, 0.0, 1.0)

    bridge_score = 1.0 - neighborhood_overlap
    bridge_score = np.maximum(bridge_score, 0.0)

    endpoint_contrast = np.abs(
        percentile - percentile.T
    )
    endpoint_contrast[~(valid & valid.T)] = 0.0

    score = (
        0.62 * mutual_preference
        + 0.23 * bridge_score * mutual_preference
        + 0.15 * endpoint_contrast * mutual_preference
    )

    score = 0.5 * (score + score.T)
    score[~(valid & valid.T)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
