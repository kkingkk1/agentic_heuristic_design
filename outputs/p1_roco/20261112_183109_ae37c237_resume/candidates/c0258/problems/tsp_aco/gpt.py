import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    positive = d[valid & (d > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        valid_values = d[valid]
        global_scale = float(np.max(valid_values))

    eps = np.finfo(float).eps
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    ordered = np.argsort(d, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(d, ordered, axis=1)

    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else best

    best = np.where(np.isfinite(best), best, global_scale)
    second = np.where(np.isfinite(second), second, best)

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    distance_term = np.exp(
        -np.clip(1.05 * relative_distance, 0.0, 60.0)
    )

    valid_counts = np.sum(valid, axis=1).astype(float)
    ranks = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_order = ordered[i]
        row_valid = valid[i, row_order]
        ranks[i, row_order[row_valid]] = np.arange(
            int(np.sum(row_valid)),
            dtype=float,
        )

    rank_scale = np.maximum(1.0, 0.75 * np.sqrt(np.maximum(valid_counts, 1.0)))
    rank_term = np.exp(-ranks / rank_scale[:, None])
    rank_term[~valid] = 0.0

    bidirectional_rank = np.sqrt(rank_term * rank_term.T)

    regret_gap = np.maximum(second - best, 0.0)
    regret = np.divide(
        regret_gap,
        np.maximum(second, global_scale),
    )
    regret = np.clip(regret, 0.0, 1.0)

    endpoint_bonus = 1.0 + 0.35 * (
        regret[:, None] + regret[None, :]
    )

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = ordered[i][valid[i, ordered[i]]]
        if candidates.size:
            nearest[i] = int(candidates[0])

    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.28, 1.0)

    scores = (
        distance_term
        * (0.62 + 0.38 * bidirectional_rank)
        * endpoint_bonus
        * mutual_bonus
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
