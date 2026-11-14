import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    finite_nonnegative &= ~diagonal

    if not np.any(finite_nonnegative):
        return scores

    # Build a conservative symmetric representation.
    forward = np.where(finite_nonnegative, distances, np.inf)
    backward = forward.T

    forward_valid = np.isfinite(forward)
    backward_valid = np.isfinite(backward)
    valid = forward_valid | backward_valid

    safe = np.full((n, n), np.inf, dtype=float)
    both = forward_valid & backward_valid
    only_forward = forward_valid & ~backward_valid
    only_backward = ~forward_valid & backward_valid

    safe[both] = np.minimum(forward[both], backward[both])
    safe[only_forward] = forward[only_forward]
    safe[only_backward] = backward[only_backward]

    np.fill_diagonal(safe, np.inf)
    valid &= ~diagonal

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    # Robust endpoint-local scales from the nearest available neighbors.
    neighbor_count = min(5, n - 1)
    ordered_distances = np.sort(safe, axis=1)
    local_values = ordered_distances[:, :neighbor_count]
    local_valid = np.isfinite(local_values)

    local_scale = np.sum(
        np.where(local_valid, local_values, 0.0),
        axis=1,
    ) / np.maximum(local_valid.sum(axis=1), 1)

    local_scale = np.where(
        local_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = np.divide(
        safe,
        pair_scale,
        out=np.full((n, n), 60.0, dtype=float),
        where=np.isfinite(safe),
    )
    normalized_distance = np.clip(normalized_distance, 0.0, 60.0)

    # Main distance signal with endpoint-adaptive scaling.
    shortness = np.exp(-1.10 * normalized_distance)

    # Endpoint-specific soft preferences.
    endpoint_ratio = np.divide(
        safe,
        local_scale[:, None],
        out=np.full((n, n), 60.0, dtype=float),
        where=np.isfinite(safe),
    )
    endpoint_ratio = np.clip(endpoint_ratio, 0.0, 60.0)

    endpoint_preference = np.exp(-endpoint_ratio)
    mutual_preference = np.sqrt(
        endpoint_preference * endpoint_preference.T
    )

    # Stable ranks and local candidate masks.
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]

    rank_decay = 1.0 / np.power(1.0 + ranks, 0.90)
    rank_consensus = np.sqrt(rank_decay * rank_decay.T)

    top_k = min(3, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        order[:, :top_k],
    ] = True
    top_neighbors &= valid

    nearest = order[:, 0]
    nearest_valid = np.isfinite(safe[np.arange(n), nearest])

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in np.flatnonzero(nearest_valid):
        j = nearest[i]
        if j != i and nearest_valid[j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    # Sparse endpoints receive a bounded compensation.
    valid_counts = valid.sum(axis=1).astype(float)
    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts, 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    # Small two-hop support for locally competitive edges.
    shared_neighbors = (
        top_neighbors.astype(float)
        @ top_neighbors.astype(float).T
    )
    shared_signal = shared_neighbors / max(float(top_k), 1.0)

    locally_competitive = (
        valid
        & (ranks < min(4, n - 1))
        & (ranks.T < min(4, n - 1))
    )

    two_hop_bonus = 1.0 + 0.08 * np.clip(
        shared_signal,
        0.0,
        1.0,
    ) * locally_competitive.astype(float)

    mutual_bonus = (
        1.0
        + 0.30 * mutual_nearest
        + 0.10 * mutual_top
    )

    scarcity_bonus = 0.86 + 0.30 * scarcity_pair

    scores = (
        shortness
        * (0.44 + 0.56 * mutual_preference)
        * (0.46 + 0.54 * rank_consensus)
        * mutual_bonus
        * two_hop_bonus
        * scarcity_bonus
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
