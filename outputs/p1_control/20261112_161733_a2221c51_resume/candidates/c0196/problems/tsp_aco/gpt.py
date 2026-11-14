import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    valid = finite_positive & finite_positive.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[finite_positive]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, eps)

    work = 0.5 * (d + d.T)
    work[~finite_positive] = replacement
    np.fill_diagonal(work, replacement)

    neighbor_count = min(2, n - 1)
    nearest = np.partition(
        work,
        neighbor_count - 1,
        axis=1,
    )[:, :neighbor_count]
    local_scale = np.maximum(
        np.mean(nearest, axis=1),
        eps,
    )

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    local_ratio = work / geometric_scale
    global_ratio = work / global_scale

    score = 1.0 / (
        1.0
        + 1.35 * local_ratio
        + 0.15 * global_ratio
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
