import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric adaptive edge-score matrix for an undirected TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    if np.any(distances[finite] < 0.0):
        raise ValueError("distance_matrix must contain non-negative distances")

    valid = finite & ~np.eye(n, dtype=bool)
    positive = valid & (distances > 0.0)

    positive_values = distances[positive]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    local_scale = np.full(n, global_scale, dtype=float)
    for i in range(n):
        row_values = distances[i, positive[i]]
        if row_values.size:
            local_scale[i] = float(np.median(row_values))

    local_scale = np.maximum(local_scale, global_scale * 1e-12)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    normalized = np.zeros((n, n), dtype=float)
    normalized[valid] = distances[valid] / np.sqrt(
        local_scale[:, None][valid] * local_scale[None, :][valid]
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-normalized[valid]) / (
        1.0 + normalized[valid]
    )

    neighborhood_mass = affinity.sum(axis=1)
    positive_mass = neighborhood_mass[neighborhood_mass > 0.0]
    mass_scale = (
        float(np.median(positive_mass))
        if positive_mass.size
        else 1.0
    )
    mass_scale = max(mass_scale, np.finfo(float).eps)

    scarcity = mass_scale / (mass_scale + neighborhood_mass)
    endpoint_bonus = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )

    scores = affinity * (0.65 + 0.35 * endpoint_bonus)
    scores = 0.5 * (scores + scores.T)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    return scores
