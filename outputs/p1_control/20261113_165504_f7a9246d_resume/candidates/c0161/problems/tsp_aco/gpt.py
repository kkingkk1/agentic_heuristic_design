import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    safe_d = np.where(valid, d, np.inf)

    positive_edges = d[valid & (d > 0.0)]
    if positive_edges.size:
        global_scale = float(np.median(positive_edges))
    else:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    k = min(4, n - 1)
    sorted_d = np.sort(safe_d, axis=1)
    neighborhood = sorted_d[:, :k]

    local_scale = np.nanmedian(
        np.where(np.isfinite(neighborhood), neighborhood, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    global_evidence = np.exp(
        -np.clip(d / (1.15 * global_scale), 0.0, 60.0)
    )
    local_evidence = np.exp(
        -np.clip(d / endpoint_scale, 0.0, 60.0)
    )

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty_like(order)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, order] = np.arange(n)[None, :]

    rank_evidence = np.zeros_like(d, dtype=float)
    rank_evidence += (ranks == 0) * 1.00
    rank_evidence += (ranks == 1) * 0.56
    rank_evidence += (ranks == 2) * 0.30
    rank_evidence += (ranks == 3) * 0.14

    reciprocal_rank = 1.0 / (1.0 + ranks.astype(float))
    reciprocal_rank = 0.5 * (reciprocal_rank + reciprocal_rank.T)

    mutual_nearest = (
        (ranks == 0)
        & (ranks.T == 0)
        & valid
        & valid.T
    ).astype(float)

    row_best = np.min(safe_d, axis=1)
    col_best = np.min(safe_d, axis=0)

    row_second = np.partition(safe_d, kth=min(1, n - 1), axis=1)[:, min(1, n - 1)]
    col_second = np.partition(safe_d, kth=min(1, n - 1), axis=0)[min(1, n - 1), :]

    row_gap = np.maximum(row_second - row_best, 0.0)
    col_gap = np.maximum(col_second - col_best, 0.0)

    row_gap_scale = (
        float(np.median(row_gap[row_gap > 0.0]))
        if np.any(row_gap > 0.0)
        else global_scale
    )
    col_gap_scale = (
        float(np.median(col_gap[col_gap > 0.0]))
        if np.any(col_gap > 0.0)
        else global_scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_regret = np.exp(
        -np.clip(
            np.maximum(d - row_best[:, None], 0.0) / row_gap_scale,
            0.0,
            60.0,
        )
    )
    col_regret = np.exp(
        -np.clip(
            np.maximum(d - col_best[None, :], 0.0) / col_gap_scale,
            0.0,
            60.0,
        )
    )
    regret = np.sqrt(row_regret * col_regret)

    savings = np.maximum(
        local_scale[:, None] + local_scale[None, :] - d,
        0.0,
    )
    savings /= local_scale[:, None] + local_scale[None, :] + eps

    unusually_long = (
        (d > 2.2 * local_scale[:, None])
        & (d > 2.2 * local_scale[None, :])
    )
    long_penalty = np.where(unusually_long, 0.55, 1.0)

    score = (
        0.27 * global_evidence
        + 0.31 * local_evidence
        + 0.16 * rank_evidence
        + 0.10 * reciprocal_rank
        + 0.08 * mutual_nearest
        + 0.05 * regret
        + 0.03 * savings
    )

    score *= long_penalty
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
