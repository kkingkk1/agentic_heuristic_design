import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
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

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.max(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 8.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    degree = np.sum(valid, axis=1)
    neighbor_count = min(4, n - 1)

    nearest_block = sorted_work[:, :neighbor_count]
    nearest_valid = np.arange(neighbor_count)[None, :] < np.minimum(
        degree[:, None],
        neighbor_count,
    )

    nearest_sum = np.sum(
        np.where(nearest_valid, nearest_block, 0.0),
        axis=1,
    )
    nearest_number = np.sum(nearest_valid, axis=1)

    local_scale = np.divide(
        nearest_sum,
        np.maximum(nearest_number, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_number > 0,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    best = sorted_work[:, 0]
    second = sorted_work[:, 1] if n > 2 else best.copy()

    best = np.where(np.isfinite(best), best, local_scale)
    second = np.where(np.isfinite(second), second, best)
    second = np.where(degree >= 2, second, best)

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    relative_distance = np.divide(
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
    )

    distance_term = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    row_ratio = np.divide(
        d,
        row_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (row_scale > 0.0),
    )
    col_ratio = np.divide(
        d,
        col_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (col_scale > 0.0),
    )

    endpoint_term = np.exp(
        -np.clip(0.45 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    rank_term = np.exp(
        -(
            ranks
            / max(1.0, 0.95 * np.sqrt(float(max(n, 2))))
        )
    )
    bidirectional_rank = np.sqrt(
        rank_term * rank_term.T
    )

    regret_term = 1.0 + 0.30 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    has_neighbor = degree > 0

    mutual_nearest = (
        has_neighbor[:, None]
        & has_neighbor[None, :]
        & (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_term = np.where(mutual_nearest, 1.25, 1.0)

    long_edge_term = np.exp(
        -0.58 * np.maximum(relative_distance - 1.0, 0.0)
    )

    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    neighborhood_mass = np.sum(proximity, axis=1)
    positive_mass = neighborhood_mass[neighborhood_mass > 0.0]

    if positive_mass.size:
        mass_reference = float(np.median(positive_mass))
    else:
        mass_reference = 1.0

    if not np.isfinite(mass_reference) or mass_reference <= 0.0:
        mass_reference = 1.0

    rarity = np.sqrt(
        mass_reference / np.maximum(neighborhood_mass, eps)
    )
    rarity = np.clip(rarity, 0.80, 1.25)
    rarity_term = np.sqrt(
        rarity[:, None] * rarity[None, :]
    )
    rarity_term = np.clip(rarity_term, 0.92, 1.08)

    profile_norm = np.sqrt(
        np.sum(proximity * proximity, axis=1)
    )
    denominator = profile_norm[:, None] * profile_norm[None, :]

    overlap = np.divide(
        proximity @ proximity.T,
        denominator,
        out=np.zeros((n, n), dtype=float),
        where=denominator > 0.0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_term = np.clip(
        1.0
        + 0.07 * (1.0 - overlap)
        - 0.025 * overlap * np.exp(
            -np.clip(relative_distance, 0.0, 40.0)
        ),
        0.94,
        1.07,
    )

    scores = (
        distance_term
        * (
            0.50
            + 0.28 * bidirectional_rank
            + 0.22 * endpoint_term
        )
        * regret_term
        * mutual_term
        * rarity_term
        * bridge_term
        * long_edge_term
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
