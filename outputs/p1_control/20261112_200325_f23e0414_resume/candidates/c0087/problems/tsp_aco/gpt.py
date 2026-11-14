import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]

    if n == 0:
        return np.zeros_like(d, dtype=float)

    if n == 1:
        return np.zeros((1, 1), dtype=float)

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

    if n == 2:
        score = np.zeros((2, 2), dtype=float)
        score[0, 1] = score[1, 0] = 1.0
        return score

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    candidate_idx = np.argpartition(work, kth=k - 1, axis=1)[:, :k]
    candidate_dist = np.take_along_axis(work, candidate_idx, axis=1)

    local_scale = np.mean(candidate_dist, axis=1)
    local_scale = np.maximum(local_scale, eps)

    first_idx = np.argmin(work, axis=1)
    first_dist = work[np.arange(n), first_idx]

    second_dist = np.partition(work, kth=1, axis=1)[:, 1]
    second_dist = np.maximum(second_dist, first_dist)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) + eps

    relative_length = d / pair_scale
    closeness = np.exp(-relative_length)

    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate_mask[rows, candidate_idx] = True
    np.fill_diagonal(candidate_mask, False)

    mutual_mask = candidate_mask & candidate_mask.T
    one_sided_mask = candidate_mask | candidate_mask.T

    candidate_strength = np.where(
        mutual_mask,
        1.0,
        np.where(one_sided_mask, 0.55, 0.0),
    )

    rank_weights = np.zeros((n, n), dtype=float)
    local_order = np.argsort(candidate_dist, axis=1)
    ordered_idx = np.take_along_axis(candidate_idx, local_order, axis=1)

    rank_values = 1.0 / (
        1.0 + np.arange(k, dtype=float)
    )

    rank_weights[rows, ordered_idx] = rank_values[None, :]
    rank_support = np.sqrt(
        rank_weights * rank_weights.T
    )

    endpoint_alternative = np.where(
        first_idx[:, None] == np.arange(n)[None, :],
        second_dist[:, None],
        first_dist[:, None],
    )

    endpoint_alternative_t = endpoint_alternative.T
    replacement_margin = (
        endpoint_alternative
        + endpoint_alternative_t
        - d
    )
    replacement_margin = np.maximum(replacement_margin, 0.0)

    indispensability = 1.0 - np.exp(
        -replacement_margin / pair_scale
    )

    score = (
        0.42 * closeness
        + 0.28 * rank_support
        + 0.18 * candidate_strength
        + 0.12 * indispensability
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    values = score[off_diagonal]

    lo = np.min(values)
    hi = np.max(values)

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
