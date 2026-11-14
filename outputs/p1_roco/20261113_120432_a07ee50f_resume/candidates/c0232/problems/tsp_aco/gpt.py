import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, scale-adaptive prior matrix for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    positive = d[finite & (d > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    reverse_finite = finite.T
    both_valid = finite & reverse_finite
    forward_only = finite & ~reverse_finite
    backward_only = reverse_finite & ~finite

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[both_valid] = 0.5 * (
        d[both_valid] + d.T[both_valid]
    )
    symmetric[forward_only] = d[forward_only]
    symmetric[backward_only] = d.T[backward_only]
    np.fill_diagonal(symmetric, np.inf)

    valid = np.isfinite(symmetric)
    if not np.any(valid):
        return result

    local_scales = np.full((n, 3), global_scale, dtype=float)
    nearest = np.full(n, global_scale, dtype=float)
    second = np.full(n, global_scale, dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        values = np.sort(symmetric[i, candidates])
        m = values.size

        nearest[i] = values[0]
        second[i] = values[1] if m > 1 else values[0]

        k1 = min(3, m)
        k2 = min(6, m)
        k3 = min(max(10, int(np.sqrt(m)) * 2), m)

        local_scales[i, 0] = np.mean(values[:k1])
        local_scales[i, 1] = np.mean(values[:k2])
        local_scales[i, 2] = np.mean(values[:k3])

    local_scales = np.where(
        np.isfinite(local_scales) & (local_scales > 0.0),
        local_scales,
        global_scale,
    )

    eps = np.finfo(float).eps

    multi_scale_shortness = np.zeros((n, n), dtype=float)
    for scale_index, weight in enumerate((0.45, 0.35, 0.20)):
        pair_scale = np.sqrt(
            np.maximum(
                np.outer(
                    local_scales[:, scale_index],
                    local_scales[:, scale_index],
                ),
                eps,
            )
        )

        relative = symmetric / pair_scale
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            component = np.exp(-np.clip(relative, 0.0, 30.0))
        multi_scale_shortness += weight * component

    order = np.argsort(symmetric, axis=1, kind="stable")
    ranks = np.full((n, n), n + 1.0, dtype=float)

    for i in range(n):
        row = order[i]
        row = row[valid[i, row]]
        ranks[i, row] = np.arange(row.size, dtype=float)

    rank_score = 1.0 / (1.0 + ranks)
    rank_consensus = np.sqrt(rank_score * rank_score.T)

    nearest_index = np.argmin(symmetric, axis=1)
    reciprocal = (
        nearest_index[:, None] == np.arange(n)[None, :]
    ) & (
        nearest_index[None, :] == np.arange(n)[:, None]
    )

    k = min(max(3, int(np.ceil(np.sqrt(max(n, 2)))), 8), max(n - 1, 1))
    candidate_mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        row = order[i]
        row = row[valid[i, row]]
        candidate_mask[i, row[:k]] = True

    shared_candidate = candidate_mask & candidate_mask.T
    one_sided_candidate = candidate_mask | candidate_mask.T

    candidate_bonus = (
        1.0
        + 0.22 * one_sided_candidate.astype(float)
        + 0.38 * shared_candidate.astype(float)
        + 0.75 * reciprocal.astype(float)
    )

    gap = np.maximum(second - nearest, 0.0)
    isolation = gap / np.maximum(nearest, global_scale)
    isolation = np.clip(isolation, 0.0, 4.0)

    scarcity = np.sqrt(
        (1.0 + 0.22 * isolation[:, None])
        * (1.0 + 0.22 * isolation[None, :])
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_to_global = symmetric / global_scale
        global_adaptation = 1.0 / (
            1.0 + 0.12 * np.clip(relative_to_global, 0.0, 20.0)
        )

    scores = (
        multi_scale_shortness
        * (0.38 + 0.62 * rank_consensus)
        * candidate_bonus
        * scarcity
        * global_adaptation
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
