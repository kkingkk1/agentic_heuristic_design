import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded symmetric edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    eps = np.finfo(float).eps

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    positive = d[finite & (d > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        valid_values = d[finite]
        global_scale = (
            float(np.median(valid_values))
            if valid_values.size
            else 1.0
        )
    global_scale = max(global_scale, eps)

    reverse_finite = finite.T
    both_valid = finite & reverse_finite
    forward_only = finite & ~reverse_finite
    backward_only = reverse_finite & ~finite

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])
    symmetric[forward_only] = d[forward_only]
    symmetric[backward_only] = d.T[backward_only]
    np.fill_diagonal(symmetric, np.inf)

    valid = np.isfinite(symmetric)
    if not np.any(valid):
        return scores

    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, global_scale, dtype=float)
    second_nearest = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = np.sort(symmetric[i, valid[i]])
        if values.size == 0:
            continue

        nearest[i] = values[0]
        second_nearest[i] = values[1] if values.size > 1 else values[0]

        k = min(3, values.size)
        positive_values = values[values > 0.0]
        if positive_values.size:
            local_scale[i] = float(np.median(positive_values[:k]))
        else:
            local_scale[i] = float(np.median(values[:k]))

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    # Mostly local normalization, with a bounded global anchor.
    anchored_scale = 0.8 * local_scale + 0.2 * global_scale
    pair_scale = np.sqrt(
        np.maximum(np.outer(anchored_scale, anchored_scale), eps)
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative = symmetric / pair_scale
        shortness = np.exp(-np.clip(relative, 0.0, 12.0))

    # Rank only valid neighbors, preserving raw distance ordering.
    ranks = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            ordered = candidates[
                np.argsort(symmetric[i, candidates], kind="stable")
            ]
            ranks[i, ordered] = np.arange(ordered.size, dtype=float)

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])
    rank_consensus = np.sqrt(rank_score * rank_score.T)

    nearest_index = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            nearest_index[i] = candidates[
                np.argmin(symmetric[i, candidates])
            ]

    reciprocal = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_index[i]
        if j >= 0 and nearest_index[j] == i:
            reciprocal[i, j] = 1.0

    # Capped scarcity: useful for locally isolated edges, but never dominant.
    gap = np.maximum(second_nearest, anchored_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        scarcity = np.clip(
            gap[:, None] / np.maximum(symmetric, eps) - 1.0,
            0.0,
            2.0,
        )
    scarcity = 0.5 * (scarcity + scarcity.T)
    scarcity[~valid] = 0.0

    # Cheap two-hop completion through a third node.
    two_hop = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in np.flatnonzero(valid[i]):
            intermediates = np.flatnonzero(valid[i] & valid[:, j])
            intermediates = intermediates[
                (intermediates != i) & (intermediates != j)
            ]

            if intermediates.size == 0:
                continue

            path_costs = (
                symmetric[i, intermediates] + symmetric[intermediates, j]
            )
            best = float(np.min(path_costs))
            scale = anchored_scale[i] + anchored_scale[j]

            if scale > 0.0:
                two_hop[i, j] = np.exp(-np.clip(best / scale, 0.0, 12.0))

    two_hop = 0.5 * (two_hop + two_hop.T)

    # Mild long-term continuation estimate from both endpoints.
    continuation = np.zeros((n, n), dtype=float)
    for i in range(n):
        valid_neighbors = np.flatnonzero(valid[i])
        if valid_neighbors.size == 0:
            continue

        cheap = np.sort(symmetric[i, valid_neighbors])
        cheap = cheap[: min(3, cheap.size)]
        endpoint_value = np.mean(
            np.exp(-np.clip(cheap / max(anchored_scale[i], eps), 0.0, 12.0))
        )
        continuation[i, valid_neighbors] = endpoint_value

    continuation = 0.5 * (continuation + continuation.T)

    # Bounded additive combination avoids excessive multiplicative distortion.
    scores = (
        0.48 * shortness
        + 0.22 * rank_consensus
        + 0.12 * two_hop
        + 0.08 * continuation
        + 0.06 * reciprocal
        + 0.04 * (0.5 * scarcity)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(scores.max(initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores
