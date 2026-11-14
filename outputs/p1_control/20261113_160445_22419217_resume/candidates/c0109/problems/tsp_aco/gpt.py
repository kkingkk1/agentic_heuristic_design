import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    safe_d = np.where(valid, d, np.inf)
    sorted_d = np.sort(safe_d, axis=1)

    k = min(5, n - 1)
    neighborhood = sorted_d[:, :k]
    finite_neighborhood = np.isfinite(neighborhood)

    if not np.any(finite_neighborhood):
        return result

    positive_values = neighborhood[(neighborhood > 0) & finite_neighborhood]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    local_scale = np.nanmedian(
        np.where(finite_neighborhood, neighborhood, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, eps)

    relative_distance = d / endpoint_scale
    global_distance = d / global_scale

    local_evidence = np.exp(-np.minimum(relative_distance, 50.0))
    global_evidence = np.exp(-np.minimum(global_distance, 50.0))

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1)
    rank_evidence = np.zeros_like(d, dtype=float)
    rank_evidence += (ranks == 0) * 1.00
    rank_evidence += (ranks == 1) * 0.55
    rank_evidence += (ranks == 2) * 0.30
    rank_evidence += (ranks == 3) * 0.15

    reciprocal_rank = 1.0 / (1.0 + ranks.astype(float))
    reciprocal_rank = 0.5 * (reciprocal_rank + reciprocal_rank.T)

    mutual_nearest = (
        (ranks == 0) & (ranks.T == 0)
    ).astype(float)

    savings = np.maximum(
        local_scale[:, None] + local_scale[None, :] - d,
        0.0,
    )
    savings /= (
        local_scale[:, None] + local_scale[None, :] + eps
    )

    endpoint_long = (
        (d > 2.2 * local_scale[:, None])
        & (d > 2.2 * local_scale[None, :])
    )
    long_penalty = np.where(endpoint_long, 0.45, 1.0)

    score = (
        0.30 * local_evidence
        + 0.24 * global_evidence
        + 0.18 * rank_evidence
        + 0.12 * reciprocal_rank
        + 0.10 * mutual_nearest
        + 0.06 * savings
    )

    score *= long_penalty
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
