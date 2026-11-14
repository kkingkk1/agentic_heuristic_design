import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric distance- and neighborhood-aware TSP edge priors."""
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
        local_scale[i] = float(np.mean(values[: min(4, values.size)]))

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

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_distance = symmetric / pair_scale
        shortness = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    order = np.argsort(symmetric, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_score = 1.0 / np.sqrt(1.0 + ranks)
    rank_consensus = np.sqrt(rank_score * rank_score.T)

    nearest_index = np.argmin(symmetric, axis=1)
    reciprocal = (
        nearest_index[:, None] == np.arange(n)[None, :]
    ) & (
        nearest_index[None, :] == np.arange(n)[:, None]
    )

    gap = np.maximum(second_nearest, global_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        scarcity = 1.0 + 0.35 * np.clip(
            gap[:, None]
            / np.maximum(symmetric, np.finfo(float).eps)
            - 1.0,
            0.0,
            2.5,
        )
    scarcity = np.sqrt(scarcity * scarcity.T)

    scores = (
        shortness
        * (0.50 + 0.50 * rank_consensus)
        * (1.0 + 0.65 * reciprocal)
        * scarcity
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
