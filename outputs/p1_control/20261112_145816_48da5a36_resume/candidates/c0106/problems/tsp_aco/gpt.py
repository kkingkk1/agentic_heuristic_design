import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        d,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)), global_scale) * 10.0

    if not np.isfinite(replacement):
        replacement = np.finfo(float).max / 4.0

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    sorted_distances = np.sort(work, axis=1)
    kth = min(2, n - 2)
    local_scale = sorted_distances[:, kth]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    normalized_length = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)

    reciprocal_rank = 1.0 / (1.0 + ranks + ranks.T)

    scale_mismatch = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )

    relative_global_length = work / global_scale

    log_score = (
        -0.95 * normalized_length
        -0.35 * relative_global_length
        + 1.65 * np.log(reciprocal_rank)
        -1.10 * scale_mismatch
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(np.where(finite_scores, log_score - maximum, -np.inf))
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
