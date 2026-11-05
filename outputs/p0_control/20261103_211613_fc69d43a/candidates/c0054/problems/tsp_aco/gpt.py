import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distance_matrix)
    if np.any(distance_matrix[finite] < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    valid = finite & ~np.eye(n, dtype=bool)
    positive = valid & (distance_matrix > 0)

    positive_distances = np.where(positive, distance_matrix, np.nan)
    scales = np.nanmedian(positive_distances, axis=1)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)

    affinity = np.zeros_like(distance_matrix, dtype=float)
    eps = np.finfo(float).eps * scales[:, None]
    np.divide(
        scales[:, None],
        distance_matrix + eps,
        out=affinity,
        where=positive,
    )

    neighborhood_mass = affinity.sum(axis=1)
    positive_mass = neighborhood_mass[neighborhood_mass > 0]
    mass_scale = np.median(positive_mass) if positive_mass.size else 1.0

    scarcity = mass_scale / (neighborhood_mass + mass_scale)
    endpoint_bonus = np.sqrt(scarcity[:, None] * scarcity[None, :])

    scores = affinity * (0.55 + 0.45 * endpoint_bonus)
    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0

    return scores
