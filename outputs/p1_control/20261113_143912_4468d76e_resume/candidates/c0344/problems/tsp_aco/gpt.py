import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(valid, d, np.inf)
    sorted_edges = np.sort(safe, axis=1)

    k = min(2, max(n - 2, 0))
    local_scale = sorted_edges[:, k]

    finite_local = np.isfinite(local_scale)
    if np.any(~finite_local):
        row_medians = np.full(n, global_scale, dtype=float)
        for i in np.flatnonzero(~finite_local):
            row_values = d[i, valid[i]]
            if row_values.size:
                row_medians[i] = float(np.median(row_values))
        local_scale = np.where(finite_local, local_scale, row_medians)

    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback_distance * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    relative_distance = safe_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    order = np.argsort(safe_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank /= max(n - 1, 1)

    mutual_rank = 0.5 * (rank + rank.T)

    endpoint_affinity = np.exp(-0.95 * np.minimum(relative_distance, 20.0))
    rank_affinity = np.exp(-2.65 * np.minimum(mutual_rank, 2.0))

    global_affinity = np.exp(
        -0.20 * np.minimum(safe_distance / global_scale, 40.0)
    )
    global_affinity = 0.70 + 0.30 * global_affinity

    local_quality = np.exp(
        -0.35 * np.minimum(local_scale / global_scale, 40.0)
    )
    accessibility = np.sqrt(
        local_quality[:, None] * local_quality[None, :]
    )
    accessibility = 0.82 + 0.18 * accessibility

    scale_gap = np.abs(local_scale[:, None] - local_scale[None, :])
    scale_total = local_scale[:, None] + local_scale[None, :] + eps
    balance = 1.0 / (1.0 + 0.65 * scale_gap / scale_total)
    balance = 0.84 + 0.16 * balance

    log_score = (
        0.34 * np.log(np.maximum(endpoint_affinity, eps))
        + 0.30 * np.log(np.maximum(rank_affinity, eps))
        + 0.18 * np.log(np.maximum(global_affinity, eps))
        + 0.10 * np.log(np.maximum(accessibility, eps))
        + 0.08 * np.log(np.maximum(balance, eps))
    )

    score = np.exp(log_score - np.max(log_score[valid]))
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
