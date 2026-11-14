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
        neginf=np.inf,
    )
    distances = np.maximum(distances, 0.0).copy()
    np.fill_diagonal(distances, np.inf)

    nearest = np.min(distances, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]

    if finite_nearest.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = max(float(np.median(finite_nearest)), np.finfo(float).eps)
    safe_nearest = np.where(np.isfinite(nearest), nearest, scale)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized = distances / scale
        desirability = 1.0 / (1.0 + normalized * normalized)

        local_scale = np.sqrt(np.outer(safe_nearest, safe_nearest))
        local_ratio = local_scale / (distances + scale)
        local_ratio = np.clip(local_ratio, 0.0, 1.0)

    nearest_index = np.argmin(distances, axis=1)
    indices = np.arange(n)

    reciprocal_nearest = (
        (nearest_index[:, None] == indices[None, :])
        & (nearest_index[None, :] == indices[:, None])
    )

    scores = desirability * (0.65 + 0.35 * local_ratio)
    scores *= 1.0 + 0.35 * reciprocal_nearest

    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(scores.max(initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores
