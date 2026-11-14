import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    # Use the better direction for asymmetric or partially missing inputs.
    safe = np.where(valid, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = safe[valid]
        global_scale = max(float(np.median(finite_values)), 1.0)
    global_scale = max(global_scale, eps)

    # Rank only valid edges. Invalid entries are placed after all valid edges.
    masked = np.where(valid, safe, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(masked, order, axis=1)
    sorted_valid = np.isfinite(sorted_distances)

    valid_count = sorted_valid.sum(axis=1)
    k = min(5, n - 1)
    neighbor_values = sorted_distances[:, :k]
    neighbor_mask = np.isfinite(neighbor_values)

    local_scale = np.divide(
        np.where(neighbor_mask, neighbor_values, 0.0).sum(axis=1),
        np.maximum(neighbor_mask.sum(axis=1), 1),
    )
    local_scale = np.where(valid_count > 0, local_scale, global_scale)
    local_scale = np.maximum(local_scale, eps)

    rank = np.full((n, n), float(n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank[~valid] = float(n)

    reverse_rank = rank.T
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized = safe / np.maximum(pair_scale, eps)

    # Distance remains the dominant signal.
    log_score = -np.clip(normalized, 0.0, 50.0)

    # Favor edges that are competitive from both endpoints.
    row_rank_score = 1.0 / np.sqrt(1.0 + rank)
    col_rank_score = 1.0 / np.sqrt(1.0 + reverse_rank)
    bidirectional_rank = np.sqrt(row_rank_score * col_rank_score)
    log_score += 0.28 * (bidirectional_rank - 1.0)

    top_k = min(3, n - 1)
    top_edges = valid & (rank < float(top_k))
    mutual_top = top_edges & top_edges.T
    log_score += 0.16 * mutual_top.astype(float)

    # Penalize edges that are much worse than the endpoint's best alternatives.
    best = np.min(np.where(valid, safe, np.inf), axis=1)
    second = np.full(n, np.inf, dtype=float)
    if n > 1:
        second_values = sorted_distances[:, 1:]
        second_valid = np.isfinite(second_values)
        second = np.min(
            np.where(second_valid, second_values, np.inf),
            axis=1,
        )
    second = np.where(np.isfinite(second), second, best)

    endpoint_best = np.minimum(best[:, None], best[None, :])
    endpoint_second = np.minimum(second[:, None], second[None, :])

    regret = np.maximum(safe - endpoint_best, 0.0)
    regret_scale = np.maximum(endpoint_second - endpoint_best, global_scale * 0.05)
    regret_penalty = regret / np.maximum(regret_scale, eps)
    log_score -= 0.035 * np.clip(regret_penalty, 0.0, 8.0)

    # Slightly compensate for nodes whose candidate choices are unusually sparse.
    sparsity = np.clip(local_scale / global_scale, 0.25, 4.0)
    sparse_pair = np.sqrt(sparsity[:, None] * sparsity[None, :])
    log_score += 0.025 * np.clip(sparse_pair - 1.0, -1.0, 2.0)

    scores = np.exp(np.clip(log_score, -60.0, 20.0))
    scores[~valid] = 0.0

    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
