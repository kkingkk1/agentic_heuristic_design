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
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid]
    global_scale = max(float(np.median(positive)), eps)

    fallback = global_scale
    safe_d = np.where(valid, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, eps)

    ranked_d = np.where(valid, d, np.inf)
    kth = min(2, n - 2)
    local_scale = np.partition(ranked_d, kth, axis=1)[:, kth]

    row_has_valid = np.any(valid, axis=1)
    local_scale = np.where(
        row_has_valid & np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, eps)

    scale_product = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = safe_d / scale_product

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks /= max(n - 1, 1)

    rank_sum = np.where(valid, ranks + ranks.T, 2.0)

    density = np.mean(
        valid & (safe_d <= 1.5 * local_scale[:, None]),
        axis=1,
    )
    density_penalty = (
        density[:, None] + density[None, :]
    ) * np.minimum(normalized_length, 2.0)

    scale_imbalance = np.abs(
        np.log(local_scale[:, None] / local_scale[None, :])
    )

    log_score = (
        -1.20 * normalized_length
        -1.70 * rank_sum
        -0.40 * (safe_d / global_scale)
        -0.25 * scale_imbalance
        -0.15 * density_penalty
    )

    finite = valid & np.isfinite(log_score)
    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    maximum = np.max(log_score[finite])
    score = np.zeros((n, n), dtype=float)
    score[finite] = np.exp(log_score[finite] - maximum)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
