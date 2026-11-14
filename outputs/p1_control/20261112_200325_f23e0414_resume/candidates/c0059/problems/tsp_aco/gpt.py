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

    if positive.size:
        fallback = float(np.max(positive))
    else:
        fallback = 1.0

    d = np.where(finite, d, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps

    k = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) + eps

    normalized_length = d / pair_scale
    short_edge = np.exp(-normalized_length)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_preference = 1.0 / (1.0 + ranks)
    mutual_preference = np.sqrt(
        rank_preference * rank_preference.T
    )

    endpoint_competition = np.minimum(
        rank_preference, rank_preference.T
    )

    third_edge_sum = (
        work[:, :, None]
        + work[None, :, :]
    )

    detours = third_edge_sum - d[:, :, None]
    candidate_mask = np.ones((n, n, n), dtype=bool)
    indices = np.arange(n)
    candidate_mask[indices, indices, :] = False
    candidate_mask[indices, :, indices] = False
    detours = np.where(candidate_mask, detours, np.inf)

    replacement_margin = np.min(detours, axis=2)
    replacement_margin = np.maximum(replacement_margin, 0.0)

    alpha_nearness = 1.0 - np.exp(
        -replacement_margin / pair_scale[:, :, None]
    )
    alpha_nearness = np.squeeze(alpha_nearness, axis=2)

    score = (
        0.46 * short_edge
        + 0.26 * mutual_preference
        + 0.18 * alpha_nearness
        + 0.10 * endpoint_competition
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    mask = ~np.eye(n, dtype=bool)
    values = score[mask]

    lo = float(np.min(values))
    hi = float(np.max(values))

    if hi > lo:
        score = (score - lo) / (hi - lo)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
