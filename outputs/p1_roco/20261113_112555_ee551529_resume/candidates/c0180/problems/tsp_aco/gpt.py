import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
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
    distances = np.maximum(distances, 0.0).copy()
    np.fill_diagonal(distances, np.inf)

    nearest = np.min(distances, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]

    if finite_nearest.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = max(
        float(np.median(finite_nearest)),
        np.finfo(float).eps,
    )

    normalized = distances / scale
    desirability = 1.0 / (1.0 + 0.9 * normalized * normalized)

    row_nearest = np.argmin(distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = row_nearest[i]
        if j != i and row_nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    row_min = np.where(np.isfinite(nearest), nearest, scale)
    row_scale = np.sqrt(np.outer(row_min, row_min))

    local_ratio = row_scale / (distances + 0.75 * scale)
    local_ratio = np.clip(local_ratio, 0.0, 1.0)

    scores = desirability * (0.45 + 0.55 * local_ratio)
    scores *= 1.0 + 0.65 * mutual

    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(scores.max(initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores
