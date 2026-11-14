import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a normalized symmetric edge-priority matrix for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)
    global_scale = max(global_scale, eps)

    forward_valid = valid
    backward_valid = valid.T

    symmetric_valid = forward_valid | backward_valid
    symmetric_distance = np.full((n, n), np.inf, dtype=float)

    both_valid = forward_valid & backward_valid
    symmetric_distance[both_valid] = (
        d[both_valid] + d.T[both_valid]
    ) * 0.5

    only_forward = forward_valid & ~backward_valid
    symmetric_distance[only_forward] = d[only_forward]

    only_backward = backward_valid & ~forward_valid
    symmetric_distance[only_backward] = d.T[only_backward]

    np.fill_diagonal(symmetric_distance, np.inf)

    usable = min(4, n - 1)
    sorted_distances = np.sort(symmetric_distance, axis=1)
    local_values = sorted_distances[:, :usable]
    local_scale = np.mean(local_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = symmetric_distance / pair_scale
    normalized_distance = np.clip(normalized_distance, 0.0, 80.0)

    affinity = np.exp(-normalized_distance)

    row_order = np.argsort(
        symmetric_distance,
        axis=1,
        kind="stable",
    )
    col_order = np.argsort(
        symmetric_distance,
        axis=0,
        kind="stable",
    )

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_affinity = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    nearest_row = np.argmin(symmetric_distance, axis=1)
    nearest_col = np.argmin(symmetric_distance, axis=0)
    indices = np.arange(n)

    mutual_nearest = (
        (nearest_row[:, None] == indices[None, :])
        & (nearest_col[None, :] == indices[:, None])
        & symmetric_valid
    ).astype(float)

    isolation_ratio = pair_scale / (
        symmetric_distance + global_scale
    )
    isolation_ratio = np.clip(isolation_ratio, 0.0, 1.0)
    isolation_bonus = 1.0 + 0.25 * isolation_ratio

    scores = (
        affinity
        * (0.45 + 0.55 * rank_affinity)
        * (1.0 + 0.35 * mutual_nearest)
        * isolation_bonus
    )

    scores[~symmetric_valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
