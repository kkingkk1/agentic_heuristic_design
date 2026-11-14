import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative local edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    valid_both = finite & finite.T
    valid_forward = finite & ~finite.T
    valid_backward = finite.T & ~finite

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid_both] = 0.5 * (
        d[valid_both] + d.T[valid_both]
    )
    symmetric[valid_forward] = d[valid_forward]
    symmetric[valid_backward] = d.T[valid_backward]
    np.fill_diagonal(symmetric, np.inf)

    valid = np.isfinite(symmetric)
    if not np.any(valid):
        return scores

    positive = symmetric[valid & (symmetric > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    local_scale = np.full(n, global_scale, dtype=float)
    degree = valid.sum(axis=1)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size:
            values = np.sort(symmetric[i, neighbors])
            local_scale[i] = float(
                np.mean(values[: min(4, values.size)])
            )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    pair_scale = np.sqrt(
        np.maximum(
            np.outer(local_scale, local_scale),
            np.finfo(float).eps,
        )
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_distance = symmetric / pair_scale
        shortness = np.exp(-np.clip(relative_distance, 0.0, 60.0))

    order = np.argsort(symmetric, axis=1, kind="stable")
    ranks = np.full((n, n), np.inf, dtype=float)

    row_index = np.arange(n)[:, None]
    rank_values = np.arange(n, dtype=float)[None, :]
    ranks[row_index, order] = rank_values

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])
    rank_consensus = np.sqrt(rank_score * rank_score.T)

    candidate_width = np.minimum(4, np.maximum(degree, 0))
    top_local = np.zeros((n, n), dtype=bool)

    for i in range(n):
        if candidate_width[i] <= 0:
            continue
        neighbors = np.flatnonzero(valid[i])
        neighbors = neighbors[
            np.argsort(symmetric[i, neighbors], kind="stable")
        ]
        top_local[i, neighbors[: candidate_width[i]]] = True

    mutual_local = top_local & top_local.T

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size:
            nearest[i] = neighbors[
                np.argmin(symmetric[i, neighbors])
            ]

    mutual_nearest = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = True

    rank_excess = np.zeros((n, n), dtype=float)
    for i in range(n):
        if degree[i] <= candidate_width[i]:
            continue

        denominator = max(
            float(degree[i] - candidate_width[i]),
            1.0,
        )
        row_excess = np.maximum(
            ranks[i] - float(candidate_width[i] - 1),
            0.0,
        ) / denominator
        rank_excess[i] = np.clip(row_excess, 0.0, 1.0)

    competitive_penalty = np.sqrt(
        (1.0 - 0.20 * rank_excess)
        * (1.0 - 0.20 * rank_excess.T)
    )

    scores = (
        shortness
        * (0.65 + 0.35 * rank_consensus)
        * (1.0 + 0.25 * mutual_local.astype(float))
        * (1.0 + 0.25 * mutual_nearest.astype(float))
        * competitive_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(scores.max(initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores
