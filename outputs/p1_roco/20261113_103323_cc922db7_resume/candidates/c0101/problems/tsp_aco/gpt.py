import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, locally scaled priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    nonnegative = np.maximum(raw, 0.0)

    forward = finite.copy()
    np.fill_diagonal(forward, False)
    backward = forward.T
    both = forward & backward

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[both] = 0.5 * (
        nonnegative[both] + nonnegative.T[both]
    )

    only_forward = forward & ~backward
    only_backward = backward & ~forward
    symmetric[only_forward] = nonnegative[only_forward]
    symmetric[only_backward] = nonnegative.T[only_backward]
    np.fill_diagonal(symmetric, np.inf)

    valid = np.isfinite(symmetric)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = symmetric[valid & (symmetric > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = symmetric[valid]
        global_scale = float(np.median(finite_values))
    global_scale = max(global_scale, eps)

    order = np.argsort(symmetric, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(symmetric, order, axis=1)

    local_width = np.empty(n, dtype=float)
    for i in range(n):
        row = sorted_distances[i]
        row = row[np.isfinite(row)]
        if row.size:
            local_width[i] = float(np.mean(row[: min(2, row.size)]))
        else:
            local_width[i] = global_scale

    local_width = np.where(
        np.isfinite(local_width) & (local_width > 0.0),
        local_width,
        global_scale,
    )
    local_width = np.maximum(local_width, eps)

    pair_width = np.sqrt(
        np.maximum(local_width[:, None] * local_width[None, :], eps)
    )

    with np.errstate(over="ignore", invalid="ignore"):
        distance_preference = np.exp(
            -np.clip(symmetric / pair_width, 0.0, 60.0)
        )

    ranks = np.zeros((n, n), dtype=float)
    nearest = np.full(n, -1, dtype=int)
    second_nearest = np.full(n, global_scale, dtype=float)

    for i in range(n):
        candidates = order[i, np.isfinite(sorted_distances[i])]
        if candidates.size == 0:
            continue

        nearest[i] = candidates[0]
        if candidates.size > 1:
            second_nearest[i] = symmetric[i, candidates[1]]
        else:
            second_nearest[i] = symmetric[i, candidates[0]]

        row_rank = np.empty(candidates.size, dtype=float)
        row_rank[:] = np.arange(candidates.size, dtype=float)
        ranks[i, candidates] = 1.0 / (1.0 + row_rank)

    rank_consensus = np.sqrt(ranks * ranks.T)

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    gap = np.maximum(second_nearest, global_scale)
    isolation = np.ones((n, n), dtype=float)

    for i in range(n):
        mask = valid[i]
        if not np.any(mask):
            continue
        ratio = gap[i] / np.maximum(symmetric[i, mask], eps) - 1.0
        isolation[i, mask] += 0.20 * np.clip(ratio, 0.0, 3.0)

    isolation = np.sqrt(isolation * isolation.T)

    scores = (
        distance_preference
        * (0.60 + 0.40 * rank_consensus)
        * (1.0 + 0.30 * mutual_nearest)
        * isolation
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float)
