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

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[finite_positive]
    fallback = max(float(np.max(positive_values)), eps)
    replacement = max(fallback * 10.0, eps)

    valid = finite_positive & finite_positive.T
    work = d.copy()
    work[~finite_positive] = replacement
    np.fill_diagonal(work, replacement)

    neighbor_count = min(2, n - 1)
    if neighbor_count == 1:
        local_scale = np.min(work, axis=1)
    else:
        nearest = np.partition(work, neighbor_count - 1, axis=1)[:, :neighbor_count]
        local_scale = np.mean(nearest, axis=1)

    local_scale = np.maximum(local_scale, eps)
    geometric_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    normalized_length = work / geometric_scale
    global_scale = max(float(np.median(positive_values)), eps)
    absolute_length = work / global_scale

    reciprocal_quality = 1.0 / (
        1.0
        + normalized_length
        + 0.5 * (work / np.maximum(local_scale[:, None], eps))
        + 0.5 * (work / np.maximum(local_scale[None, :], eps))
    )

    log_score = (
        -0.85 * normalized_length
        -0.18 * absolute_length
        + 2.4 * np.log(np.maximum(reciprocal_quality, eps))
    )

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
