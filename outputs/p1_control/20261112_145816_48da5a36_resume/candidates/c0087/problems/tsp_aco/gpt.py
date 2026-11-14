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

    positive_values = d[valid]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0:
        return np.zeros((n, n), dtype=float)

    safe_d = np.where(valid, d, fallback * 10.0)
    eps = np.finfo(float).eps
    safe_d = np.maximum(safe_d, eps)

    row_sorted = np.sort(safe_d, axis=1)
    kth = min(2, n - 2)
    local_scale = row_sorted[:, kth]

    missing_scale = ~np.isfinite(local_scale) | (local_scale <= 0)
    if np.any(missing_scale):
        replacement = float(np.median(positive_values))
        local_scale[missing_scale] = max(replacement, eps)

    geometric_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )
    relative_length = safe_d / geometric_scale

    order = np.argsort(safe_d, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    normalized_rank = rank / max(n - 1, 1)

    reciprocal_signal = 1.0 / (1.0 + relative_length)
    rank_signal = np.exp(-2.4 * (normalized_rank + normalized_rank.T))

    neighbor_count = min(3, n - 1)
    near = rank <= neighbor_count
    mutual_signal = (near & near.T).astype(float)

    global_scale = float(np.median(positive_values))
    global_scale = max(global_scale, eps)
    global_signal = 1.0 / (1.0 + safe_d / global_scale)

    log_density_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    density_balance = 1.0 / (1.0 + log_density_gap)

    base = (
        0.46 * reciprocal_signal
        + 0.24 * rank_signal
        + 0.16 * mutual_signal
        + 0.14 * global_signal
    )

    # 保留跨密度连接的最低权重，减轻局部稠密区域的过度聚集。
    base *= 0.72 + 0.28 * density_balance

    # 小幅保留较远边的全局对比度，避免评分完全塌缩到局部近邻。
    contrast = safe_d / (safe_d + global_scale)
    base += 0.035 * contrast

    score = np.where(valid, base, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
