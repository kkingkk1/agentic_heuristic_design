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
    else:
        global_scale = float(np.max(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    penalty = max(float(np.max(finite_values)), global_scale) * 8.0
    penalty += global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_work = np.take_along_axis(work, order, axis=1)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    degree = valid.sum(axis=1)
    k = min(4, n - 1)

    nearest = sorted_work[:, :k]
    nearest_valid = np.arange(k)[None, :] < np.minimum(
        degree[:, None], k
    )
    nearest_sum = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    )
    nearest_count = nearest_valid.sum(axis=1)

    local_scale = np.divide(
        nearest_sum,
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
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

    scale_product = local_scale[:, None] * local_scale[None, :]
    geometric_scale = np.sqrt(scale_product)

    relative_distance = np.divide(
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
    )

    distance_quality = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    row_ratio = np.divide(
        d,
        local_scale[:, None],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        local_scale[None, :],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    endpoint_quality = np.exp(
        -np.clip(0.46 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    rank_quality = np.exp(
        -ranks / max(1.0, 0.95 * np.sqrt(float(max(n, 2))))
    )
    bidirectional_rank = np.sqrt(
        rank_quality * rank_quality.T
    )

    regret_factor = 1.0 + 0.30 * (
        regret[:, None] + regret[None, :]
    )

    nearest_node = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (degree[:, None] > 0)
        & (degree[None, :] > 0)
        & (nearest_node[:, None] == indices[None, :])
        & (nearest_node[None, :] == indices[:, None])
    )
    mutual_factor = np.where(mutual_nearest, 1.18, 1.0)

    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    neighborhood_mass = proximity.sum(axis=1)
    positive_mass = neighborhood_mass[neighborhood_mass > 0.0]
    mass_reference = (
        float(np.median(positive_mass))
        if positive_mass.size
        else 1.0
    )
    if not np.isfinite(mass_reference) or mass_reference <= 0.0:
        mass_reference = 1.0

    rarity = np.sqrt(
        mass_reference / np.maximum(neighborhood_mass, eps)
    )
    rarity = np.clip(rarity, 0.82, 1.22)
    rarity_factor = np.sqrt(
        rarity[:, None] * rarity[None, :]
    )
    rarity_factor = np.clip(rarity_factor, 0.92, 1.08)

    profile_norm = np.sqrt(np.sum(proximity * proximity, axis=1))
    profile_denominator = (
        profile_norm[:, None] * profile_norm[None, :]
    )
    overlap = np.divide(
        proximity @ proximity.T,
        profile_denominator,
        out=np.zeros((n, n), dtype=float),
        where=profile_denominator > 0.0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_factor = np.clip(
        1.0
        + 0.06 * (1.0 - overlap)
        - 0.02 * overlap * np.exp(
            -np.clip(relative_distance, 0.0, 40.0)
        ),
        0.94,
        1.06,
    )

    long_edge_factor = np.exp(
        -0.56 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_quality
        * (
            0.52
            + 0.28 * bidirectional_rank
            + 0.20 * endpoint_quality
        )
        * regret_factor
        * mutual_factor
        * rarity_factor
        * bridge_factor
        * long_edge_factor
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
