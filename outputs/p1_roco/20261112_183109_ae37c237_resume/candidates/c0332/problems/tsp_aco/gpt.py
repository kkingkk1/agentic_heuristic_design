import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    finite_values = symmetric_distance[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.median(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    ordered = np.sort(symmetric_distance, axis=1)
    k = min(4, n - 1)
    neighbors = ordered[:, :k]
    neighbor_valid = np.isfinite(neighbors)

    neighbor_sum = np.sum(
        np.where(neighbor_valid, neighbors, 0.0),
        axis=1,
    )
    neighbor_count = np.sum(neighbor_valid, axis=1)

    local_scale = np.divide(
        neighbor_sum,
        np.maximum(neighbor_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=neighbor_count > 0,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    best = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, global_scale)
    safe_second = np.where(np.isfinite(second), second, safe_best)

    regret = np.divide(
        safe_second - safe_best,
        np.maximum(safe_second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(safe_second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    relative_distance = np.divide(
        symmetric_distance,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    row_ratio = np.divide(
        symmetric_distance,
        row_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        symmetric_distance,
        col_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    distance_quality = np.exp(
        -np.clip(1.10 * relative_distance, 0.0, 60.0)
    )

    endpoint_quality = np.exp(
        -np.clip(0.36 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    rank_order = np.argsort(
        np.where(valid, symmetric_distance, np.inf),
        axis=1,
        kind="stable",
    )
    rank_positions = np.empty((n, n), dtype=float)
    rank_positions[
        np.arange(n)[:, None],
        rank_order,
    ] = np.arange(n, dtype=float)

    rank_quality = np.exp(
        -np.clip(
            0.055
            * (rank_positions + rank_positions.T),
            0.0,
            30.0,
        )
    )

    neighborhood_kernel = np.exp(
        -np.clip(symmetric_distance / row_scale, 0.0, 24.0)
    )
    neighborhood_kernel[~valid] = 0.0
    np.fill_diagonal(neighborhood_kernel, 0.0)

    neighborhood_mass = neighborhood_kernel.sum(axis=1)
    positive_mass = neighborhood_mass[neighborhood_mass > 0.0]
    mass_reference = (
        float(np.median(positive_mass))
        if positive_mass.size
        else 1.0
    )

    if not np.isfinite(mass_reference) or mass_reference <= 0.0:
        mass_reference = 1.0

    scarcity = np.sqrt(
        mass_reference / np.maximum(neighborhood_mass, eps)
    )
    scarcity = np.clip(scarcity, 0.90, 1.12)

    scarcity_quality = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    urgency = 1.0 + 0.16 * (
        regret[:, None] + regret[None, :]
    )

    nearest = rank_order[:, 0]
    indices = np.arange(n)

    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.16, 1.0)

    long_edge_penalty = np.exp(
        -0.42 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (
            0.60
            + 0.22 * endpoint_quality
            + 0.18 * rank_quality
        )
        * scarcity_quality
        * urgency
        * mutual_bonus
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
