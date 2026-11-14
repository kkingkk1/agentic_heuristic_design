import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0)
    valid = valid & valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    fallback = float(np.median(positive_values))
    fallback = max(fallback, np.finfo(float).eps)

    safe_d = np.where(valid, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, np.finfo(float).eps)

    sorted_d = np.sort(np.where(valid, d, np.inf), axis=1)
    kth = min(2, n - 2)
    local_scale = sorted_d[:, kth]

    row_has_valid = np.any(valid, axis=1)
    local_scale = np.where(row_has_valid, local_scale, fallback)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback,
    )

    normalized_length = safe_d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)

    finite_rank = np.where(valid, ranks, 1.0)
    rank_sum = finite_rank + finite_rank.T

    global_scale = max(float(np.median(positive_values)), np.finfo(float).eps)

    local_density = np.mean(
        valid & (safe_d <= 1.5 * local_scale[:, None]),
        axis=1,
    )
    density_penalty = (
        local_density[:, None] + local_density[None, :]
    ) * np.minimum(normalized_length, 2.0)

    scale_balance = np.abs(
        np.log(local_scale[:, None] / local_scale[None, :])
    )

    log_score = (
        -1.15 * normalized_length
        -1.8 * rank_sum
        -0.45 * safe_d / global_scale
        -0.25 * scale_balance
        -0.20 * density_penalty
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = np.max(log_score[finite_scores])
    score = np.exp(np.where(finite_scores, log_score - maximum, -np.inf))
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
