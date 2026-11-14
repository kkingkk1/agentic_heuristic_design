import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)

    finite_positive = np.isfinite(d) & (d > 0)
    np.fill_diagonal(finite_positive, False)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[finite_positive]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, eps)

    work = np.where(finite_positive, d, replacement)
    np.fill_diagonal(work, replacement)

    k = min(2, n - 1)
    nearest = np.partition(work, k - 1, axis=1)[:, :k]

    if k == 1:
        local_scale = np.maximum(nearest[:, 0], eps)
    else:
        local_scale = np.maximum(
            0.8 * nearest[:, 0] + 0.2 * nearest[:, 1],
            eps,
        )

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_length = work / geometric_scale
    absolute_length = work / global_scale

    score = 1.0 / (
        1.0
        + 1.3 * normalized_length
        + 0.04 * absolute_length
    )

    score[~finite_positive] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
