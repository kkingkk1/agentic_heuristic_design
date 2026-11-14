import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    forward = np.where(finite_nonnegative, distances, np.inf)
    backward = np.where(finite_nonnegative.T, distances.T, np.inf)

    both = np.isfinite(forward) & np.isfinite(backward)
    distance = np.where(
        both,
        0.5 * (forward + backward),
        np.minimum(forward, backward),
    )
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(finite_values)), 1.0)
    )
    scale = max(scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    sorted_valid = np.isfinite(sorted_distance)

    neighbor_limit = min(max(3, int(np.sqrt(n)) + 1), n - 1)
    local_neighbors = sorted_distance[:, :neighbor_limit]
    local_mask = np.isfinite(local_neighbors)

    local_mean = np.sum(
        np.where(local_mask, local_neighbors, 0.0),
        axis=1,
    ) / np.maximum(np.sum(local_mask, axis=1), 1)

    local_mean = np.where(
        np.isfinite(local_mean),
        local_mean,
        scale,
    )
    local_mean = np.maximum(local_mean, eps)

    pair_local = np.sqrt(
        local_mean[:, None] * local_mean[None, :]
    )

    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    valid_counts = np.sum(valid, axis=1).astype(float)
    relative_rank = 1.0 - (
        row_rank / np.maximum(valid_counts[:, None], 1.0)
    )
    relative_rank = np.clip(relative_rank, 0.0, 1.0)

    relative_distance = distance / np.maximum(local_mean[:, None], eps)
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    endpoint_closeness = np.sqrt(
        np.exp(-relative_distance)
        * np.exp(-relative_distance.T)
    )

    endpoint_competitiveness = np.sqrt(
        relative_rank * relative_rank.T
    )

    nearest = np.take_along_axis(
        distance,
        order[:, :min(3, n - 1)],
        axis=1,
    )
    nearest = np.where(np.isfinite(nearest), nearest, local_mean[:, None])

    best_alternative = nearest[:, 0]
    second_alternative = nearest[:, 1] if nearest.shape[1] > 1 else nearest[:, 0]

    edge_gap = np.maximum(second_alternative[:, None] - distance, 0.0)
    edge_gap_t = np.maximum(second_alternative[None, :] - distance, 0.0)

    scarcity = np.sqrt(
        (1.0 + edge_gap / np.maximum(local_mean[:, None], eps))
        * (1.0 + edge_gap_t / np.maximum(local_mean[None, :], eps))
    )
    scarcity = np.clip(scarcity, 1.0, 3.0)

    if n > 2:
        via = distance[:, :, None] + distance[None, :, :]
        via[:, np.arange(n), :] = np.inf
        via[:, :, np.arange(n)] = np.inf

        best_detour = np.min(via, axis=2)
        detour_valid = np.isfinite(best_detour) & valid

        detour_efficiency = np.ones((n, n), dtype=float)
        detour_efficiency[detour_valid] = np.clip(
            best_detour[detour_valid]
            / np.maximum(distance[detour_valid], eps),
            1.0,
            8.0,
        )
    else:
        detour_efficiency = np.ones((n, n), dtype=float)

    strength = np.where(
        valid,
        np.exp(-np.clip(distance / np.maximum(pair_local, eps), 0.0, 50.0)),
        0.0,
    )

    total_strength = np.sum(strength, axis=1)
    strongest_strength = np.max(strength, axis=1)
    concentration = strongest_strength / np.maximum(total_strength, eps)

    balance = 1.0 - np.sqrt(
        concentration[:, None] * concentration[None, :]
    )
    balance = np.clip(balance, 0.0, 1.0)

    scores = (
        endpoint_closeness
        * (0.20 + 0.80 * endpoint_competitiveness)
        * (0.72 + 0.28 * np.clip(scarcity / 3.0, 0.0, 1.0))
        * (0.84 + 0.16 * np.clip(detour_efficiency / 8.0, 0.0, 1.0))
        * (0.94 + 0.10 * balance)
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
