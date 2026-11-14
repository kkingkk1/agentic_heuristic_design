import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, distance-dominant edge-priority matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    # Treat an edge as usable when at least one direction is available.
    safe = np.minimum(
        np.where(directed_valid, distances, np.inf),
        np.where(directed_valid.T, distances.T, np.inf),
    )
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)
    global_scale = max(global_scale, eps)

    valid_counts = valid.sum(axis=1)
    row_order = np.argsort(safe, axis=1, kind="stable")

    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    neighborhood_size = min(4, n - 1)
    local_scale = np.empty(n, dtype=float)

    for i in range(n):
        row_values = safe[i, row_order[i]]
        row_values = row_values[np.isfinite(row_values)]
        if row_values.size:
            local_scale[i] = float(
                np.mean(row_values[:neighborhood_size])
            )
        else:
            local_scale[i] = global_scale

    local_scale = np.maximum(local_scale, eps)
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    # Main signal: preserve raw distance ordering under local scaling.
    normalized_distance = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    # Mild bidirectional local-rank support.
    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )
    rank_factor = 0.72 + 0.28 * rank_score

    top_k = min(3, n - 1)
    top_neighbors = valid & (row_rank < float(top_k))
    mutual_top = top_neighbors & top_neighbors.T

    # Explicit mutual-nearest-neighbor signal with bounded influence.
    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        valid_positions = row_order[i][np.isfinite(safe[i, row_order[i]])]
        if valid_positions.size:
            nearest[i] = int(valid_positions[0])

    indices = np.arange(n)
    mutual_nearest = (
        valid
        & (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )

    mutual_factor = (
        1.0
        + 0.18 * mutual_nearest.astype(float)
        + 0.06 * mutual_top.astype(float)
    )

    # Weak scarcity correction: protect useful alternatives near sparse endpoints.
    endpoint_best = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    opportunity_ratio = safe / np.maximum(endpoint_best, eps)
    scarcity_factor = 1.0 + 0.08 * np.clip(
        opportunity_ratio - 1.0,
        0.0,
        2.0,
    )

    # Mild concentration control, applied only to locally competitive edges.
    popularity = top_neighbors.astype(float).sum(axis=0)
    popularity_reference = max(float(np.mean(popularity)), 1.0)
    concentration = np.clip(
        popularity / popularity_reference,
        0.0,
        3.0,
    )
    concentration_factor = 1.0 / np.sqrt(
        (1.0 + 0.07 * concentration[:, None])
        * (1.0 + 0.07 * concentration[None, :])
    )
    concentration_factor = np.where(
        top_neighbors | top_neighbors.T,
        concentration_factor,
        1.0,
    )

    scores = (
        distance_score
        * rank_factor
        * mutual_factor
        * scarcity_factor
        * concentration_factor
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
