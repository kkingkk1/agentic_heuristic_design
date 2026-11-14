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
    directed_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(directed_valid):
        return np.zeros((n, n), dtype=float)

    reverse = distances.T
    reverse_valid = np.isfinite(reverse) & (reverse >= 0.0) & ~diagonal

    valid = directed_valid | reverse_valid
    safe = np.where(directed_valid, distances, np.inf)
    safe = np.minimum(safe, np.where(reverse_valid, reverse, np.inf))
    safe = np.where(valid, safe, np.inf)
    np.fill_diagonal(safe, np.inf)

    finite_values = safe[np.isfinite(safe)]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = max(float(np.median(finite_values)), 1.0)
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    local_scales = np.empty(n, dtype=float)
    nearest_indices = []

    for i in range(n):
        row = safe[i]
        finite_row = row[np.isfinite(row)]

        if finite_row.size == 0:
            local_scales[i] = global_scale
            nearest_indices.append(np.array([], dtype=int))
            continue

        k = min(5, finite_row.size)
        order = np.argsort(row, kind="stable")
        nearest = order[:k]
        nearest_indices.append(nearest)

        local_value = float(np.mean(row[nearest]))
        local_scales[i] = max(local_value, np.finfo(float).eps)

    pair_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    normalized_distance = safe / pair_scale
    endpoint_distance = safe / np.maximum(
        np.minimum(local_scales[:, None], local_scales[None, :]),
        np.finfo(float).eps,
    )

    distance_affinity = 1.0 / (1.0 + np.clip(normalized_distance, 0.0, 1e12))
    endpoint_affinity = 1.0 / (1.0 + np.clip(endpoint_distance, 0.0, 1e12))

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )
    reciprocal_rank = np.clip(reciprocal_rank, 0.0, 1.0)

    top_k = min(3, max(1, n - 1))
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    nearest = np.argmin(safe, axis=1)
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    base = (
        0.54 * distance_affinity
        + 0.28 * endpoint_affinity
        + 0.18 * reciprocal_rank
    )

    bonuses = (
        0.08 * mutual_top
        + 0.12 * mutual_nearest
    )

    scores = base + bonuses
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
