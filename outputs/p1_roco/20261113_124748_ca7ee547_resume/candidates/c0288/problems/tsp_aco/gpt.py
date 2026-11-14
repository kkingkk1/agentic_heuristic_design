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
    valid_input = finite & (distances >= 0.0) & ~diagonal

    if not np.any(valid_input):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = safe[valid]
        global_scale = float(np.median(finite_values)) if finite_values.size else 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    degree = np.sum(valid, axis=1)
    neighbor_count = np.maximum(degree, 1)

    row_values = np.where(valid, safe, np.inf)
    usable_k = min(5, max(n - 1, 1))

    nearest_values = np.partition(
        row_values,
        kth=min(usable_k - 1, n - 1),
        axis=1,
    )[:, :usable_k]

    local_scale = np.sum(
        np.where(np.isfinite(nearest_values), nearest_values, 0.0),
        axis=1,
    ) / neighbor_count
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = safe / np.maximum(pair_scale, np.finfo(float).eps)

    global_score = np.exp(
        -np.clip((safe / global_scale) ** 1.12, 0.0, 50.0)
    )
    local_score = np.exp(
        -np.clip(relative_distance ** 1.28, 0.0, 50.0)
    )

    endpoint_score = np.sqrt(
        np.exp(
            -np.clip(
                (safe / local_scale[:, None]) ** 1.10,
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                (safe / local_scale[None, :]) ** 1.10,
                0.0,
                50.0,
            )
        )
    )

    row_order = np.argsort(row_values, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    rank_score = 1.0 / np.sqrt(
        (1.0 + rank) * (1.0 + rank.T)
    )

    top_k = min(4, n - 1)
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

    nearest = row_order[:, 0]
    mutual_nearest = (
        valid
        & (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    ).astype(float)

    row_best = np.min(row_values, axis=1)
    row_second = np.partition(
        row_values,
        kth=min(1, n - 1),
        axis=1,
    )[:, min(1, n - 1)]

    row_gap = np.maximum(row_second - row_best, 0.0)
    row_gap_scale = np.maximum(local_scale, global_scale * 0.25)
    endpoint_need = 1.0 + 0.10 * np.clip(
        row_gap / row_gap_scale,
        0.0,
        1.0,
    )

    endpoint_need_pair = np.sqrt(
        endpoint_need[:, None] * endpoint_need[None, :]
    )

    local_density = np.clip(global_scale / local_scale, 0.25, 4.0)
    density_pair = np.sqrt(
        local_density[:, None] * local_density[None, :]
    )
    density_adjustment = 1.0 + 0.08 * np.tanh(
        np.log(np.maximum(density_pair, np.finfo(float).eps))
    )

    scores = (
        (0.48 * global_score + 0.52 * local_score)
        * endpoint_score
        * (0.30 + 0.70 * rank_score)
        * endpoint_need_pair
        * density_adjustment
        * (
            1.0
            + 0.42 * mutual_nearest
            + 0.24 * mutual_top
        )
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
