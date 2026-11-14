import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    distances = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    distances = np.maximum(distances, 0.0)
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)

    nearest = np.min(distances, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]

    if finite_nearest.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(finite_nearest))
    scale = max(scale, np.finfo(float).eps)

    safe_distances = np.where(np.isfinite(distances), distances, scale)

    normalized = safe_distances / scale
    inverse_score = 1.0 / (1.0 + normalized * normalized)

    endpoint_scale = np.sqrt(
        np.maximum(nearest[:, None], scale)
        * np.maximum(nearest[None, :], scale)
    )
    relative_score = endpoint_scale / (
        endpoint_scale + safe_distances + 0.5 * scale
    )
    relative_score = np.clip(relative_score, 0.0, 1.0)

    nearest_index = np.argmin(distances, axis=1)
    mutual = (
        nearest_index[:, None] == np.arange(n)[None, :]
    ) & (
        nearest_index[None, :] == np.arange(n)[:, None]
    )
    mutual = mutual.astype(float)

    scores = inverse_score * (0.55 + 0.45 * relative_score)
    scores *= 1.0 + 0.60 * mutual

    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
