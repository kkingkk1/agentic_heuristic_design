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
    usable = finite & ~np.eye(n, dtype=bool)

    if not np.any(usable):
        return scores

    distances = np.where(finite, np.maximum(distance_matrix, 0.0), 0.0)
    distances = 0.5 * (distances + distances.T)
    valid = np.isfinite(distances) & ~np.eye(n, dtype=bool)

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    affinity = np.zeros_like(distances, dtype=float)
    affinity[valid] = np.exp(-distances[valid] / scale)

    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)

    degree = np.sum(affinity, axis=1)
    laplacian = np.diag(degree) - affinity

    try:
        laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    except TypeError:
        laplacian_pinv = np.linalg.pinv(laplacian)

    resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance = 0.5 * (resistance + resistance.T)

    positive_resistance = resistance[valid & (resistance > 0.0)]
    resistance_scale = (
        float(np.median(positive_resistance))
        if positive_resistance.size
        else 1.0
    )
    resistance_scale = max(resistance_scale, np.finfo(float).eps)

    structural_bonus = resistance / resistance_scale
    structural_bonus = structural_bonus / (1.0 + structural_bonus)

    scores = affinity * (0.5 + structural_bonus)
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
