import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & (distances >= 0.0)
    finite &= ~diagonal

    symmetric_count = finite.astype(float) + finite.T.astype(float)
    edge_valid = symmetric_count > 0.0
    edge_valid &= ~diagonal

    if not np.any(edge_valid):
        return np.zeros((n, n), dtype=float)

    values = np.where(finite, distances, 0.0)
    d = (values + values.T) / np.maximum(symmetric_count, 1.0)

    valid_values = d[edge_valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        fallback = float(np.max(positive_values))
        global_scale = float(np.median(positive_values))
    else:
        fallback = 1.0
        global_scale = 1.0

    fallback = max(fallback, np.finfo(float).eps)
    global_scale = max(global_scale, np.finfo(float).eps)

    d = np.where(edge_valid, d, fallback)
    d[diagonal] = np.inf

    neighbor_count = max(1, min(n - 1, int(np.sqrt(n)) + 1))
    ordered = np.sort(d, axis=1)

    weights = np.linspace(1.0, 0.6, neighbor_count)
    weights /= np.sum(weights)
    local_neighbors = ordered[:, :neighbor_count]
    local_scale = local_neighbors @ weights
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / np.maximum(pair_scale, np.finfo(float).eps)

    row_order = np.argsort(d, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    rank_scale = max(float(n - 1), 1.0)
    bidirectional_rank = (
        row_rank + row_rank.T
    ) / (2.0 * rank_scale)

    nearest = ordered[:, 0]
    if n > 2:
        second_nearest = ordered[:, 1]
    else:
        second_nearest = nearest

    local_gap = np.maximum(
        second_nearest - nearest,
        0.05 * local_scale,
    )
    local_gap = np.maximum(local_gap, np.finfo(float).eps)

    advantage_i = np.clip(
        (second_nearest[:, None] - d) / local_gap[:, None],
        0.0,
        1.0,
    )
    advantage_j = np.clip(
        (second_nearest[None, :] - d) / local_gap[None, :],
        0.0,
        1.0,
    )
    mutual_advantage = np.sqrt(advantage_i * advantage_j)

    endpoint_relative = 0.5 * (
        d / np.maximum(local_scale[:, None], np.finfo(float).eps)
        + d / np.maximum(local_scale[None, :], np.finfo(float).eps)
    )

    log_score = (
        -0.88 * np.clip(relative_distance ** 0.78, 0.0, 50.0)
        -0.28 * np.clip(bidirectional_rank, 0.0, 50.0)
        -0.20 * np.clip(endpoint_relative ** 0.65, 0.0, 50.0)
        -0.10 * np.clip(d / global_scale, 0.0, 50.0)
        +0.24 * mutual_advantage
    )

    log_score[~edge_valid] = -np.inf
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    reference = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - reference, -50.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
