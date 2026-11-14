import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return stochastic edge-priority indicators for a TSP distance matrix."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(d)
    nonnegative = d >= 0
    directed_valid = finite & nonnegative
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid & directed_valid.T
    pair_d = np.where(valid, 0.5 * (d + d.T), np.inf)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    safe = pair_d

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=int)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n)[None, :]
    ranks = np.where(valid, ranks, n)

    nearest = np.min(safe, axis=1)
    positive_edges = pair_d[valid & (pair_d > 0)]
    positive_nearest = nearest[np.isfinite(nearest) & (nearest > 0)]

    if positive_nearest.size:
        global_scale = float(np.median(positive_nearest))
    elif positive_edges.size:
        global_scale = float(np.median(positive_edges))
    else:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    local_scale = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    global_score = np.exp(
        -np.minimum(np.divide(safe, global_scale), 40.0)
    )
    local_score = np.exp(
        -np.minimum(np.divide(safe, pair_scale), 40.0)
    )

    rank_score = np.exp(
        -0.58 * (
            ranks.astype(float)[:, :] + ranks.astype(float).T
        )
    )
    rank_score = np.where(valid, rank_score, 0.0)

    nearest_mask = valid & (ranks == 0)
    mutual_nearest = nearest_mask & nearest_mask.T

    second = np.full(n, np.inf, dtype=float)
    if n >= 3:
        second = safe[np.arange(n), order[:, 1]]

    has_alternatives = (
        np.isfinite(nearest)
        & np.isfinite(second)
        & (second > 0)
    )
    regret_vertex = np.zeros(n, dtype=float)
    regret_vertex[has_alternatives] = np.clip(
        (second[has_alternatives] - nearest[has_alternatives])
        / np.maximum(second[has_alternatives], eps),
        0.0,
        1.0,
    )

    selective_regret = np.where(
        regret_vertex >= 0.15,
        regret_vertex,
        0.0,
    )
    regret_edge = nearest_mask * 0.5 * (
        selective_regret[:, None] + selective_regret[None, :]
    )

    candidate = valid & (ranks <= min(3, n - 1))
    endpoint_excess = np.maximum(
        safe / np.maximum(local_scale[:, None], eps)
        + safe / np.maximum(local_scale[None, :], eps)
        - 2.0,
        0.0,
    )
    plausibility = np.exp(-np.minimum(0.35 * endpoint_excess, 8.0))
    plausibility = np.where(valid, plausibility, 0.0)

    bridge = np.zeros((n, n), dtype=float)
    visited = np.zeros(n, dtype=bool)

    for root in range(n):
        if visited[root]:
            continue

        best = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        best[root] = 0.0

        for _ in range(n):
            choices = np.where(~visited, best, np.inf)
            u = int(np.argmin(choices))

            if not np.isfinite(choices[u]):
                break

            visited[u] = True

            if parent[u] >= 0:
                v = parent[u]
                bridge[u, v] = 1.0
                bridge[v, u] = 1.0

            better = (
                (~visited)
                & valid[u]
                & (pair_d[u] < best)
            )
            best[better] = pair_d[u, better]
            parent[better] = u

    score = (
        0.28 * global_score
        + 0.30 * local_score
        + 0.18 * rank_score
        + 0.08 * regret_edge
        + 0.10 * mutual_nearest.astype(float)
        + 0.06 * bridge
    )

    score *= plausibility
    score = np.where(candidate, score, 0.82 * score)
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    positive = score[valid & (score > 0)]
    if positive.size:
        floor = max(float(np.percentile(positive, 10)) * 0.02, eps)
        score[valid] = np.maximum(score[valid], floor)

    valid_values = score[valid]
    mean_score = float(np.mean(valid_values)) if valid_values.size else 0.0

    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
