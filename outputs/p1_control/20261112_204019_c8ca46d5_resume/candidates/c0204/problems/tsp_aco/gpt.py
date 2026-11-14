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

    off_diagonal = ~np.eye(n, dtype=bool)
    work = d.copy()
    np.fill_diagonal(work, np.inf)

    eps = np.finfo(float).eps

    # Slightly wider neighborhoods stabilize local scales and reduce
    # over-reinforcement of only the very closest edges.
    k = max(1, min(n - 1, int(np.ceil(1.75 * np.sqrt(n)))))
    order = np.argsort(work, axis=1, kind="mergesort")
    neighbors = order[:, :k]

    local_scale = np.take_along_axis(work, neighbors, axis=1).mean(axis=1)
    local_scale = np.maximum(local_scale, eps)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :]) + eps

    relative_length = d / pair_scale
    distance_attraction = np.exp(-relative_length)

    # Bidirectional competition: an edge is preferred when it competes well
    # against alternatives at both endpoints.
    competition = np.exp(-relative_length)
    row_total = competition.sum(axis=1, keepdims=True) - 1.0
    row_total = np.maximum(row_total, eps)
    directional_competition = competition / row_total
    mutual_competition = np.sqrt(
        directional_competition * directional_competition.T
    )

    # Reciprocal rank gives a cheap rank-based compatibility signal.
    ranks = np.empty_like(order, dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    base_score = (
        distance_attraction
        * np.sqrt(np.maximum(mutual_competition, 0.0))
        * mutual_rank
    )

    # Sparse two-hop alternative estimate. Each intermediate node updates
    # only the edges touching its own local neighborhood, avoiding an O(n^3)
    # detour tensor while retaining local replacement information.
    candidate_width = max(2, min(n - 1, int(np.ceil(2.5 * np.sqrt(n)))))
    candidate_order = order[:, :candidate_width]

    replacement = np.full((n, n), np.inf, dtype=float)

    for intermediate in range(n):
        left = candidate_order[intermediate]
        right = candidate_order[intermediate]

        left_cost = work[:, intermediate][:, None]
        right_cost = work[intermediate, :][None, :]
        detour = left_cost + right_cost

        active = np.zeros((n, n), dtype=bool)
        active[np.ix_(left, right)] = True
        active[intermediate, :] = False
        active[:, intermediate] = False

        replacement = np.minimum(
            replacement,
            np.where(active, detour, np.inf),
        )

    finite_replacement = np.isfinite(replacement)
    margin = np.where(
        finite_replacement,
        np.maximum(replacement - d, 0.0),
        0.0,
    )
    normalized_margin = margin / pair_scale

    # Keep only mutually plausible candidate edges. The wider candidate
    # set prevents brittle one-neighbor decisions.
    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_mask[row_index, candidate_order] = True
    mutual_candidate = candidate_mask & candidate_mask.T & off_diagonal

    margin_signal = 1.0 - np.exp(-normalized_margin)
    margin_signal = np.minimum(margin_signal, 1.0)

    # Conservative reward: alternative-margin information is subordinate to
    # the distance and competition terms.
    bonus = 0.10 * margin_signal * mutual_candidate.astype(float)

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
