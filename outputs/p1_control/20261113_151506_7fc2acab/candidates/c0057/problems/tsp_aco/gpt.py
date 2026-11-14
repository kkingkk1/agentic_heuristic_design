import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for a TSP distance matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distances, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    finite_edges = d[np.isfinite(d)]
    positive_edges = finite_edges[finite_edges > 0]

    eps = np.finfo(float).eps
    global_scale = (
        float(np.median(positive_edges)) if positive_edges.size else 1.0
    )
    global_scale = max(global_scale, eps)

    # Adaptive local scale based on a moderate nearest-neighbor quantile.
    neighbor_count = min(
        n - 1,
        max(2, int(np.ceil(np.sqrt(n)))),
    )
    sorted_d = np.sort(d, axis=1)
    local_scale = sorted_d[:, neighbor_count - 1]

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    # Smooth local closeness with a robust global distance correction.
    local_closeness = np.exp(-d / pair_scale)
    global_closeness = 1.0 / (1.0 + d / global_scale)

    local_closeness[~np.isfinite(local_closeness)] = 0.0
    global_closeness[~np.isfinite(global_closeness)] = 0.0

    # Reciprocal rank preference from both endpoints.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)
    np.fill_diagonal(rank_score, 0.0)

    mutual_rank = np.sqrt(rank_score * rank_score.T)
    mutual_closeness = np.sqrt(local_closeness * local_closeness.T)

    # Endpoint competition: an edge is stronger when it is short relative
    # to the other alternatives available to both endpoints.
    row_best = np.min(d, axis=1)
    row_best = np.where(np.isfinite(row_best), row_best, global_scale)

    relative_distance = d / np.maximum(row_best[:, None], eps)
    competition = 1.0 / (1.0 + relative_distance)
    competition[~np.isfinite(competition)] = 0.0
    mutual_competition = np.sqrt(competition * competition.T)

    # Clarke-Wright savings using each endpoint's strongest local connection.
    savings = row_best[:, None] + row_best[None, :] - d
    savings = np.maximum(savings, 0.0)

    positive_savings = savings[savings > 0]
    savings_scale = (
        float(np.median(positive_savings))
        if positive_savings.size
        else global_scale
    )
    savings_scale = max(savings_scale, eps)
    savings_score = np.tanh(savings / savings_scale)
    savings_score[~np.isfinite(savings_score)] = 0.0

    # Shared-neighbor support rewards edges whose endpoints have compatible
    # local neighborhoods, while retaining direct distance information.
    normalized = np.exp(-d / np.maximum(local_scale[:, None], eps))
    normalized[~np.isfinite(normalized)] = 0.0
    shared_neighbor = normalized @ normalized.T
    shared_neighbor = shared_neighbor / np.maximum(
        np.sqrt(
            np.sum(normalized * normalized, axis=1)[:, None]
            * np.sum(normalized * normalized, axis=1)[None, :]
        ),
        eps,
    )
    shared_neighbor[~np.isfinite(shared_neighbor)] = 0.0
    np.fill_diagonal(shared_neighbor, 0.0)

    score = (
        0.34 * mutual_closeness
        + 0.22 * mutual_rank
        + 0.16 * mutual_competition
        + 0.12 * global_closeness
        + 0.10 * savings_score
        + 0.06 * shared_neighbor
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    # Sharpen useful distinctions without making the distribution too sparse.
    score = np.maximum(score, 0.0)
    score = np.power(score, 1.25)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    final_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        final_max,
        out=np.zeros_like(score),
        where=final_max > 0,
    )

    return score.astype(float, copy=False)
