import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d)
    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps
    k = max(1, min(n - 1, int(np.ceil(1.25 * np.sqrt(n)))))

    nearest = np.partition(work, k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)
    scale = np.sqrt(local_scale[:, None] * local_scale[None, :]) + eps

    relative_length = d / scale
    closeness = np.exp(-relative_length)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    endpoint_competition = 0.5 * (
        np.sqrt(rank_score * rank_score.T) + mutual_rank
    )
    endpoint_competition = np.clip(endpoint_competition, 0.0, 1.0)

    base_score = (
        closeness
        * mutual_rank
        * (1.0 + 0.12 * endpoint_competition)
    )

    detours = (
        work[:, None, :]
        + work[None, :, :]
        - d[:, :, None]
    )

    node_index = np.arange(n)
    valid_intermediate = (
        (node_index[None, None, :] != node_index[:, None, None])
        & (node_index[None, None, :] != node_index[None, :, None])
    )
    detours = np.where(valid_intermediate, detours, np.inf)

    replacement_margin = np.min(detours, axis=2)
    replacement_margin = np.where(
        np.isfinite(replacement_margin),
        np.maximum(replacement_margin, 0.0),
        0.0,
    )
    normalized_margin = replacement_margin / scale

    off_diagonal = ~np.eye(n, dtype=bool)

    local_order = np.argsort(
        -normalized_margin,
        axis=1,
        kind="mergesort",
    )
    local_rank = np.empty_like(local_order, dtype=float)
    local_rank[row_index, local_order] = np.arange(n, dtype=float)[None, :]

    candidate_count = max(1, int(np.ceil(0.20 * (n - 1))))
    local_high = (
        (local_rank < candidate_count)
        & off_diagonal
    )

    indispensable = 1.0 - np.exp(-normalized_margin)
    mutual_high = local_high & local_high.T

    bonus_strength = 0.08
    bonus = np.where(
        mutual_high,
        bonus_strength * indispensable * endpoint_competition,
        0.0,
    )

    score = base_score * (1.0 + bonus)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    values = score[off_diagonal]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    lo = float(np.min(values))
    hi = float(np.max(values))

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
