import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    if not np.all(finite):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distances must be non-negative")

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    eps = max(np.finfo(float).eps, np.median(positive) * 1e-12) if positive.size else 1e-12

    sorted_rows = np.sort(d, axis=1)
    k = min(max(2, n // 4), n - 1)
    local_scale = np.mean(sorted_rows[:, :k], axis=1)
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = d / endpoint_scale

    closeness = np.exp(-normalized_distance)
    inverse_distance = 1.0 / (d + eps)

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    row_max = np.maximum(row_max, eps)
    row_preference = inverse_distance / row_max

    mutual_preference = np.sqrt(row_preference * row_preference.T)

    nearest = np.argmin(d, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    score = (
        0.45 * closeness
        + 0.35 * mutual_preference
        + 0.15 * np.sqrt(row_preference * row_preference.T)
        + 0.05 * mutual_nearest
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return 0.5 * (score + score.T)
