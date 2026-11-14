import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
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

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d > 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    fallback = float(np.median(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    eps = np.finfo(float).eps

    safe_d = np.where(valid, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, eps)

    if n > 2:
        local_scale = np.partition(safe_d, 1, axis=1)[:, 1]
    else:
        local_scale = safe_d[:, 1]

    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = safe_d / np.maximum(geometric_scale, eps)

    short_edge_score = np.exp(-0.85 * normalized_distance)

    scale_sum = local_scale[:, None] + local_scale[None, :]
    scale_ratio = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / np.maximum(scale_sum, eps)

    scale_compatibility = np.exp(-0.20 * scale_ratio)

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values

    valid_count = np.maximum(valid.sum(axis=1), 1)
    normalized_ranks = ranks / valid_count[:, None]

    endpoint_rank = np.exp(
        -1.8 * (normalized_ranks + normalized_ranks.T)
    )

    neighbor_limit = np.maximum(
        3.0,
        np.ceil(0.20 * valid_count.astype(float)),
    )
    near_endpoint = (
        (ranks <= neighbor_limit[:, None])
        & valid
    )
    at_least_one_near = near_endpoint | near_endpoint.T

    short_and_promising = (
        (normalized_distance <= 2.25)
        & at_least_one_near
        & valid
    )

    near_bonus = np.where(short_and_promising, 1.12, 1.0)

    score = (
        short_edge_score
        * endpoint_rank
        * scale_compatibility
        * near_bonus
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
