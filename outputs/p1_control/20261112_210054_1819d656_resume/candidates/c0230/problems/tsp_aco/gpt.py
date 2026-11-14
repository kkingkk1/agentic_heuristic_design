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

    local_k = max(
        1,
        min(n - 1, int(np.ceil(1.25 * np.sqrt(n)))),
    )

    nearest = np.argpartition(
        work,
        kth=local_k - 1,
        axis=1,
    )[:, :local_k]

    local_scale = np.mean(
        np.take_along_axis(work, nearest, axis=1),
        axis=1,
    )
    local_scale = np.maximum(local_scale, eps)

    scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) + eps

    relative_length = d / scale
    closeness = np.exp(-relative_length)

    order = np.argsort(work, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)

    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)
    base_score = closeness * mutual_rank

    replacement_margin = np.full((n, n), np.inf, dtype=float)
    node_ids = np.arange(n)

    for neighbor_slot in range(local_k):
        candidates = nearest[:, neighbor_slot]

        source_leg = d[node_ids, candidates]
        source_detours = (
            source_leg[:, None]
            + d[candidates, :]
            - d
        )
        source_detours[node_ids, candidates] = np.inf
        replacement_margin = np.minimum(
            replacement_margin,
            source_detours,
        )

        target_leg = d[candidates, node_ids]
        target_detours = (
            d[:, candidates]
            + target_leg[None, :]
            - d
        )
        target_detours[candidates, node_ids] = np.inf
        replacement_margin = np.minimum(
            replacement_margin,
            target_detours,
        )

    replacement_margin = np.where(
        np.isfinite(replacement_margin),
        np.maximum(replacement_margin, 0.0),
        0.0,
    )

    normalized_margin = replacement_margin / scale

    off_diagonal = ~np.eye(n, dtype=bool)

    candidate_count = max(
        1,
        int(np.ceil(0.25 * (n - 1))),
    )

    high_threshold = np.partition(
        normalized_margin,
        kth=max(0, n - candidate_count),
        axis=1,
    )[:, max(0, n - candidate_count)][:, None]

    local_high = (
        (normalized_margin >= high_threshold)
        & off_diagonal
    )

    indispensable = 1.0 - np.exp(-normalized_margin)
    mutual_high = local_high & local_high.T

    bonus_strength = 0.08
    bonus = np.where(
        mutual_high,
        bonus_strength * indispensable,
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
