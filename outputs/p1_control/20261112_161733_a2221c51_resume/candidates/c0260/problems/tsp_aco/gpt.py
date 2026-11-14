import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw) & (raw >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    d = np.where(valid, 0.5 * (raw + raw.T), np.inf)
    eps = np.finfo(float).eps

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    positive_work = np.where(valid & (d > 0.0), d, np.inf)

    if n > 1:
        nearest_two = np.partition(positive_work, kth=1, axis=1)[:, :2]
        first_scale = nearest_two[:, 0]
        second_scale = nearest_two[:, 1]
    else:
        first_scale = np.full(n, np.inf)
        second_scale = np.full(n, np.inf)

    local_scale = np.where(
        np.isfinite(second_scale),
        second_scale,
        np.where(np.isfinite(first_scale), first_scale, global_scale),
    )
    local_scale = np.maximum(local_scale, eps)

    positive_safe = np.maximum(d, eps)

    normalized_length = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = np.clip(
        np.where(valid, normalized_length, 0.0),
        0.0,
        1e6,
    )

    global_quality = np.clip(
        global_scale / positive_safe,
        0.0,
        1e12,
    )

    local_quality_i = 1.0 / (
        1.0 + d / np.maximum(local_scale[:, None], eps)
    )
    local_quality_j = 1.0 / (
        1.0 + d / np.maximum(local_scale[None, :], eps)
    )
    bilateral_quality = np.sqrt(
        np.maximum(local_quality_i, 0.0)
        * np.maximum(local_quality_j, 0.0)
    )

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        row = np.where(valid[i], d[i], np.inf)
        if np.isfinite(row).any():
            nearest[i] = int(np.argmin(row))

    mutual_nearest = np.zeros((n, n), dtype=float)
    valid_nearest = np.flatnonzero(nearest >= 0)
    for i in valid_nearest:
        j = nearest[i]
        if nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    scale_mismatch = np.abs(
        np.log(
            np.maximum(local_scale[:, None], eps)
            / np.maximum(local_scale[None, :], eps)
        )
    )
    scale_mismatch = np.minimum(scale_mismatch, 12.0)

    log_score = (
        0.72 * np.log(np.maximum(global_quality, eps))
        + 1.05 * np.log(np.maximum(bilateral_quality, eps))
        - 0.78 * normalized_length
        + 0.24 * mutual_nearest
        - 0.06 * scale_mismatch
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not finite_scores.any():
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.clip(log_score - maximum, -745.0, 0.0))
    score = np.where(valid, score, 0.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
