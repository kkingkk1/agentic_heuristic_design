import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    d = np.where(finite, d, 0.0)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d) & (d > 0.0)
    valid &= valid.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid]
    global_scale = max(float(np.median(positive)), eps)
    fallback = max(float(np.mean(positive)), global_scale)

    safe = np.where(valid, d, fallback * 4.0)
    safe = np.maximum(safe, eps)

    row_counts = valid.sum(axis=1)
    row_median = np.full(n, fallback, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        if values.size:
            row_median[i] = float(np.median(values))

    row_median = np.maximum(row_median, eps)

    deviation = np.abs(
        np.where(valid, d, row_median[:, None]) - row_median[:, None]
    )
    row_mad = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = deviation[i, valid[i]]
        if values.size:
            row_mad[i] = float(np.median(values))

    bandwidth = np.maximum(
        1.4826 * row_mad + 0.25 * row_median,
        eps,
    )

    centered = (safe - row_median[:, None]) / bandwidth[:, None]
    kernel = np.exp(-0.5 * np.minimum(centered * centered, 100.0))
    kernel[~valid] = 0.0

    row_mass = kernel.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)
    transition = kernel / row_mass[:, None]

    accessibility = transition.sum(axis=0)
    accessibility = np.maximum(accessibility, eps)
    hub_penalty = 1.0 / np.sqrt(accessibility[:, None] * accessibility[None, :])

    score = np.sqrt(
        np.maximum(transition, eps)
        * np.maximum(transition.T, eps)
    )
    score *= hub_penalty
    score *= np.sqrt(
        row_counts[:, None] / np.maximum(row_counts[:, None] + row_counts[None, :], 1)
        * row_counts[None, :] / np.maximum(row_counts[:, None] + row_counts[None, :], 1)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
