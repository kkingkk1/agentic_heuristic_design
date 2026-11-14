import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(distances) & (distances > 0.0)
    valid &= ~np.eye(n, dtype=bool)

    symmetric_valid = valid & valid.T
    if not np.any(symmetric_valid):
        return score

    d = 0.5 * (distances + distances.T)
    valid = np.isfinite(d) & (d > 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    positive = d[valid]
    global_scale = float(np.median(positive))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    working = np.where(valid, d, np.inf)
    np.fill_diagonal(working, np.inf)

    k = min(7, n - 1)
    nearest_values = np.partition(working, kth=k - 1, axis=1)[:, :k]
    local_scale = np.median(nearest_values, axis=1)

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    relative_distance = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = np.where(valid, relative_distance, np.inf)

    order = np.argsort(working, axis=1, kind="stable")
    ranks = np.full((n, n), np.inf, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 1.15 * np.sqrt(float(n)))
    rank_strength = np.exp(-ranks / rank_scale)
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )
    bidirectional_rank = np.nan_to_num(
        bidirectional_rank,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    nearest = order[:, 0]
    mutual_nearest = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = int(nearest[i])
        if (
            j != i
            and j < n
            and valid[i, j]
            and nearest[j] == i
            and valid[j, i]
        ):
            mutual_nearest[i, j] = True
            mutual_nearest[j, i] = True

    second_values = np.partition(working, kth=min(1, n - 2), axis=1)
    best = second_values[:, 0]
    second = second_values[:, 1] if n > 2 else best

    best = np.where(np.isfinite(best), best, global_scale)
    second = np.where(np.isfinite(second), second, best)

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=second > 0.0,
    )
    regret = np.clip(regret, 0.0, 1.0)

    endpoint_regret = 1.0 + 0.35 * (
        regret[:, None] + regret[None, :]
    )

    distance_term = np.exp(
        -0.85 * np.clip(relative_distance, 0.0, 60.0)
    )

    long_edge_penalty = np.exp(
        -0.35 * np.maximum(relative_distance - 1.0, 0.0)
    )

    score = (
        distance_term
        * (0.58 + 0.42 * bidirectional_rank)
        * endpoint_regret
        * long_edge_penalty
        * (1.0 + 0.45 * mutual_nearest)
    )

    score = np.where(valid, score, 0.0)
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
    ).astype(float, copy=False)
