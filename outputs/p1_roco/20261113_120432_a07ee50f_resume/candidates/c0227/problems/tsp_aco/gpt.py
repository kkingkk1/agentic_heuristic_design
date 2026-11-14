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
    reference_values = positive if positive.size else d[finite]
    global_scale = float(np.median(reference_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    forward = np.where(finite, d, np.nan)
    backward = forward.T

    both = np.isfinite(forward) & np.isfinite(backward)
    one_sided = np.isfinite(forward) ^ np.isfinite(backward)

    safe = np.full((n, n), np.inf, dtype=float)
    safe[both] = 0.5 * (forward[both] + backward[both])
    safe[one_sided] = np.where(
        np.isfinite(forward[one_sided]),
        forward[one_sided],
        backward[one_sided],
    )

    valid = np.isfinite(safe) & (safe >= 0.0)
    np.fill_diagonal(valid, False)
    np.fill_diagonal(safe, np.inf)

    if not np.any(valid):
        return result

    valid_counts = valid.sum(axis=1).astype(float)
    work = np.where(valid, safe, np.inf)

    k = min(6, n - 1)
    nearest = np.partition(work, kth=k - 1, axis=1)[:, :k]
    nearest = np.where(np.isfinite(nearest), nearest, np.nan)

    local_scale = np.nanmedian(nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 1e-3)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], np.finfo(float).eps)
    )

    relative_distance = safe / pair_scale
    relative_distance = np.nan_to_num(
        relative_distance,
        nan=50.0,
        posinf=50.0,
        neginf=0.0,
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    short_edge = np.exp(-1.35 * np.power(relative_distance, 0.85))

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    row_rank = ranks + 1.0
    rank_score = 1.0 / np.power(row_rank, 0.82)

    bilateral_rank = np.sqrt(rank_score * rank_score.T)
    rank_gap = np.abs(np.log1p(ranks) - np.log1p(ranks.T))
    rank_agreement = np.exp(-0.55 * rank_gap)

    candidate_width = min(5, n - 1)
    candidate_mask = ranks < candidate_width
    mutual_candidate = candidate_mask & candidate_mask.T

    nearest_index = np.argmin(work, axis=1)
    reciprocal = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest_index[i]
        if np.isfinite(work[i, j]) and nearest_index[j] == i:
            reciprocal[i, j] = 1.0
            reciprocal[j, i] = 1.0

    degree_factor = 1.0 / np.sqrt(np.maximum(valid_counts, 1.0))
    degree_factor /= max(float(np.max(degree_factor)), np.finfo(float).eps)

    scarcity = 0.5 * (degree_factor[:, None] + degree_factor[None, :])

    local_competition = np.sum(
        np.exp(-safe / np.maximum(local_scale[:, None], global_scale * 1e-3)),
        axis=1,
    )
    local_competition = np.where(
        np.isfinite(local_competition) & (local_competition > 0.0),
        local_competition,
        1.0,
    )
    competition_factor = 1.0 / np.sqrt(local_competition)
    competition_factor /= max(
        float(np.max(competition_factor)),
        np.finfo(float).eps,
    )
    competition_pair = 0.5 * (
        competition_factor[:, None] + competition_factor[None, :]
    )

    score = (
        short_edge
        * (0.35 + 0.65 * bilateral_rank)
        * (0.45 + 0.55 * rank_agreement)
        * (0.86 + 0.14 * mutual_candidate.astype(float))
        * (1.0 + 0.28 * reciprocal)
        * (0.82 + 0.30 * scarcity)
        * (0.88 + 0.20 * competition_pair)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float)
