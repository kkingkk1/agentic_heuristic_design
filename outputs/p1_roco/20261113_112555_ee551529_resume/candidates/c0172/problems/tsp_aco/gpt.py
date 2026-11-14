import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive symmetric edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    if not np.any(finite):
        return result

    positive = d[finite & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else float(np.median(d[finite]))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    # Safe symmetric distance construction.
    forward = np.where(finite, d, np.nan)
    backward = forward.T

    both = np.isfinite(forward) & np.isfinite(backward)
    only_forward = np.isfinite(forward) & ~np.isfinite(backward)
    only_backward = ~np.isfinite(forward) & np.isfinite(backward)

    safe = np.full((n, n), scale * 1e6, dtype=float)
    safe[both] = 0.5 * (forward[both] + backward[both])
    safe[only_forward] = forward[only_forward]
    safe[only_backward] = backward[only_backward]

    valid = both | only_forward | only_backward
    np.fill_diagonal(valid, False)
    np.fill_diagonal(safe, np.inf)

    degree = valid.sum(axis=1)
    if not np.any(valid):
        return result

    work = np.where(valid, safe, np.inf)
    k = min(max(3, int(np.sqrt(max(n, 2)))), 8, n - 1)

    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    finite_nearest = np.isfinite(nearest)

    local_scale = np.sum(np.where(finite_nearest, nearest, 0.0), axis=1)
    local_scale /= np.maximum(finite_nearest.sum(axis=1), 1)

    positive_local = local_scale[(local_scale > 0.0) & np.isfinite(local_scale)]
    fallback = float(np.median(positive_local)) if positive_local.size else scale

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, fallback * 0.05)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], np.finfo(float).eps)
    )

    # Adaptive Gaussian shortness gives smooth contrast across density regions.
    normalized_distance = np.divide(
        safe,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(safe),
    )
    shortness = np.exp(-0.5 * np.square(np.minimum(normalized_distance, 12.0)))

    # Stable endpoint-relative ranks.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=np.float64)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]

    rank_fraction = ranks / np.maximum(degree[:, None] - 1.0, 1.0)
    rank_weight = np.exp(-1.8 * np.minimum(rank_fraction, 1.0))

    bilateral_preference = np.sqrt(
        np.maximum(rank_weight * rank_weight.T, 0.0)
    )

    # Reward edges that are competitive from both endpoints.
    candidate_threshold = np.minimum(k - 1, np.maximum(degree - 1, 0))
    candidate = (
        (ranks <= candidate_threshold[:, None])
        & (ranks.T <= candidate_threshold[None, :])
        & valid
    )

    one_sided_candidate = (
        (ranks <= candidate_threshold[:, None])
        | (ranks.T <= candidate_threshold[None, :])
    ) & valid

    candidate_score = 1.0 + 0.35 * candidate.astype(float)
    candidate_score += 0.10 * one_sided_candidate.astype(float)

    # Reciprocal nearest-neighbor structure captures locally stable edges.
    nearest_index = np.argmin(work, axis=1)
    reciprocal = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest_index[i]
        if valid[i, j] and nearest_index[j] == i:
            reciprocal[i, j] = 1.0
            reciprocal[j, i] = 1.0

    # Scarcity protects nodes with few alternatives.
    scarcity = 1.0 / np.sqrt(np.maximum(degree.astype(float), 1.0))
    scarcity_pair = 0.5 * (scarcity[:, None] + scarcity[None, :])
    scarcity_pair /= max(float(np.max(scarcity_pair)), np.finfo(float).eps)

    score = (
        shortness
        * (0.55 + 0.45 * bilateral_preference)
        * candidate_score
        * (1.0 + 0.30 * reciprocal)
        * (0.88 + 0.24 * scarcity_pair)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float)
