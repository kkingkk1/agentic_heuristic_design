import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(d)

    if not np.all(finite | diagonal):
        raise ValueError(
            "distance_matrix contains invalid non-finite off-diagonal values"
        )

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(1.0, fallback)

    k = max(1, min(n - 1, int(np.sqrt(n))))
    candidate_width = max(k, min(n - 1, 2 * k))

    candidate_idx = np.argpartition(
        work, candidate_width - 1, axis=1
    )[:, :candidate_width]
    candidate_dist = np.take_along_axis(work, candidate_idx, axis=1)

    nearest_positive = np.where(
        np.isfinite(candidate_dist) & (candidate_dist > eps),
        candidate_dist,
        np.nan,
    )
    local_scale = np.nanmedian(nearest_positive, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, eps)

    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate_mask[rows, candidate_idx] = True
    candidate_mask |= candidate_mask.T
    np.fill_diagonal(candidate_mask, False)

    local_rank = np.full((n, n), np.inf, dtype=float)

    order = np.argsort(candidate_dist, axis=1, kind="stable")
    ordered_idx = np.take_along_axis(candidate_idx, order, axis=1)
    ordered_dist = np.take_along_axis(candidate_dist, order, axis=1)

    ordered_rank = np.arange(candidate_width, dtype=float)
    ordered_rank = ordered_rank[None, :] / max(1.0, candidate_width - 1)

    local_rank[rows, ordered_idx] = ordered_rank
    rank_sum = local_rank + local_rank.T

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized_distance = work / (pair_scale + eps)

    locality = 1.0 / (1.0 + normalized_distance)

    finite_rank_sum = np.where(np.isfinite(rank_sum), rank_sum, 2.0)
    competition = 1.0 / (1.0 + 1.75 * finite_rank_sum)

    reciprocal = (
        np.isfinite(local_rank)
        & np.isfinite(local_rank.T)
    ).astype(float)

    scale_ratio = np.maximum(
        local_scale[:, None] / (local_scale[None, :] + eps),
        local_scale[None, :] / (local_scale[:, None] + eps),
    )
    bridge_strength = (scale_ratio - 1.0) / (scale_ratio + 1.0)
    bridge = 1.0 + 0.28 * bridge_strength * np.exp(
        -0.35 * np.minimum(normalized_distance, 20.0)
    )

    score = locality
    score *= 0.72 + 0.28 * competition
    score *= 1.0 + 0.24 * reciprocal
    score *= bridge
    score[~candidate_mask] = 0.0
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
