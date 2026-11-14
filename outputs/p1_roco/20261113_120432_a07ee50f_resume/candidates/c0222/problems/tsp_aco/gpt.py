import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge-priority scores for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    finite_nonnegative = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(finite_nonnegative):
        return result

    positive_values = distances[
        finite_nonnegative & (distances > 0.0)
    ]
    all_values = distances[finite_nonnegative]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(all_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    # Build a safe symmetric graph. A one-sided finite edge remains usable.
    safe = np.where(finite_nonnegative, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe) & ~diagonal
    if not np.any(valid):
        return result

    # Estimate each endpoint's scale from its nearest valid neighbors.
    neighborhood_size = min(5, n - 1)
    nearest_values = np.partition(
        safe,
        kth=neighborhood_size - 1,
        axis=1,
    )[:, :neighborhood_size]

    nearest_valid = np.isfinite(nearest_values)
    neighbor_counts = nearest_valid.sum(axis=1)

    local_scale = (
        np.sum(
            np.where(nearest_valid, nearest_values, 0.0),
            axis=1,
        )
        / np.maximum(neighbor_counts, 1)
    )

    local_scale = np.where(
        neighbor_counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            eps,
        )
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_distance = safe / pair_scale
        endpoint_row_distance = safe / local_scale[:, None]
        endpoint_col_distance = safe / local_scale[None, :]

    distance_score = np.exp(
        -1.05 * np.clip(relative_distance, 0.0, 50.0)
    )

    endpoint_affinity = np.sqrt(
        np.exp(-np.clip(endpoint_row_distance, 0.0, 50.0))
        * np.exp(-np.clip(endpoint_col_distance, 0.0, 50.0))
    )

    # Stable row and column rankings provide two-sided compatibility evidence.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

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

    rank_decay_row = 1.0 / np.power(1.0 + row_rank, 0.88)
    rank_decay_col = 1.0 / np.power(1.0 + col_rank, 0.88)

    rank_consensus = np.sqrt(
        rank_decay_row * rank_decay_col
    )
    rank_consensus[~valid] = 0.0

    nearest_row = row_order[:, 0]
    nearest_col = col_order[0, :]

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    ).astype(float)

    top_k = min(2, n - 1)
    row_top = np.zeros((n, n), dtype=bool)
    col_top = np.zeros((n, n), dtype=bool)

    row_top[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True

    col_top[
        col_order[:top_k, :],
        np.arange(n)[None, :],
    ] = True

    mutual_top = (
        row_top
        & col_top
        & valid
    ).astype(float)

    # Mild compensation for endpoints with fewer usable alternatives.
    valid_counts = valid.sum(axis=1).astype(float)
    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts, 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    isolation_ratio = pair_scale / (
        safe + global_scale
    )
    isolation_bonus = 1.0 + 0.16 * np.clip(
        isolation_ratio,
        0.0,
        1.0,
    )

    scores = (
        distance_score
        * endpoint_affinity
        * (0.46 + 0.54 * rank_consensus)
        * (
            1.0
            + 0.34 * mutual_nearest
            + 0.12 * mutual_top
        )
        * (0.90 + 0.20 * scarcity_pair)
        * isolation_bonus
    )

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
