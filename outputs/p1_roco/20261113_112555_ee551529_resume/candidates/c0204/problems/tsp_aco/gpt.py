import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, locally validated distance-based TSP edge priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    positive = d[finite & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

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
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        values = np.sort(symmetric[i, candidates])
        nearest[i] = values[0]
        second_nearest[i] = values[1] if values.size > 1 else values[0]

        local_count = min(4, values.size)
        local_values = values[:local_count]
        positive_local = local_values[local_values > 0.0]
        if positive_local.size:
            local_scale[i] = float(np.median(positive_local))
        else:
            local_scale[i] = global_scale

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

    ranks = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            ordered = candidates[
                np.argsort(symmetric[i, candidates], kind="stable")
            ]
            ranks[i, ordered] = np.arange(ordered.size, dtype=float)

    finite_ranks = np.where(np.isfinite(ranks), ranks, 0.0)
    rank_score = 1.0 / (1.0 + finite_ranks)

    endpoint_competitive = np.zeros((n, n), dtype=bool)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            top_count = min(max(3, int(np.ceil(np.sqrt(candidates.size)))), candidates.size)
            ordered = candidates[
                np.argsort(symmetric[i, candidates], kind="stable")
            ]
            endpoint_competitive[i, ordered[:top_count]] = True

    locally_validated = endpoint_competitive | endpoint_competitive.T
    mutual_competitive = endpoint_competitive & endpoint_competitive.T

    nearest_index = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            nearest_index[i] = candidates[
                np.argmin(symmetric[i, candidates])
            ]

    mutual_nearest = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = nearest_index[i]
        if j >= 0 and nearest_index[j] == i:
            mutual_nearest[i, j] = True

    rank_consensus = np.sqrt(
        np.clip(rank_score * rank_score.T, 0.0, 1.0)
    )

    local_gap = np.maximum(second_nearest - nearest, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        opportunity = local_gap[:, None] / (
            np.maximum(symmetric, 0.0) + local_scale[:, None]
        )
    opportunity = np.clip(np.nan_to_num(opportunity, nan=0.0), 0.0, 1.5)
    opportunity = np.sqrt(opportunity * opportunity.T)

    global_shortness = np.exp(
        -np.clip(symmetric / global_scale, 0.0, 60.0)
    )
    global_bonus = 1.0 + 0.15 * global_shortness * locally_validated

    local_bonus = (
        0.72
        + 0.28 * rank_consensus
        + 0.16 * locally_validated
        + 0.22 * mutual_competitive
        + 0.28 * mutual_nearest
    )

    scores = (
        shortness
        * local_bonus
        * global_bonus
        * (1.0 + 0.18 * opportunity * locally_validated)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(scores.max(initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores
