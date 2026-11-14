import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded, symmetric edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)
    valid &= valid.T

    if not np.any(valid):
        return scores

    with np.errstate(all="ignore"):
        symmetric_distance = 0.5 * (
            distances + distances.T
        )

    finite_values = symmetric_distance[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

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
    invalid_penalty = 6.0 * max(finite_max, global_scale) + global_scale

    work = np.where(valid, symmetric_distance, invalid_penalty)
    np.fill_diagonal(work, invalid_penalty)

    order = np.argsort(work, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(work, order, axis=1)

    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else best.copy()

    best = np.where(np.isfinite(best), best, global_scale)
    second = np.where(np.isfinite(second), second, best)

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        symmetric_distance,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
    )

    distance_quality = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    row_ratio = np.divide(
        symmetric_distance,
        local_scale[:, None],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        symmetric_distance,
        local_scale[None, :],
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    endpoint_quality = np.exp(
        -np.clip(0.5 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    rank_scale = max(1.0, 0.9 * np.sqrt(float(n)))
    rank_quality = np.exp(
        -0.5
        * (
            order.argsort(axis=1).astype(float)
            / rank_scale
            + order.argsort(axis=1).T.astype(float)
            / rank_scale
        )
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)
    regret_bonus = 1.0 + 0.24 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.18, 1.0)

    neighbor_count = min(
        n - 1,
        max(2, int(np.ceil(np.sqrt(float(n))))),
    )

    neighbor_mask = np.zeros((n, n), dtype=bool)
    neighbor_mask[
        np.arange(n)[:, None],
        order[:, :neighbor_count],
    ] = True
    neighbor_mask &= valid

    local_density = neighbor_mask.sum(axis=1).astype(float)
    density_reference = float(
        np.median(local_density[local_density > 0.0])
    )
    if not np.isfinite(density_reference) or density_reference <= 0.0:
        density_reference = 1.0

    scarcity = np.sqrt(
        density_reference / np.maximum(local_density, 1.0)
    )
    scarcity = np.clip(scarcity, 0.86, 1.18)
    scarcity_term = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    shared_neighbors = (
        neighbor_mask.astype(float)
        @ neighbor_mask.astype(float).T
    )
    overlap = np.divide(
        shared_neighbors,
        float(max(1, neighbor_count)),
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    bridge_signal = np.clip(
        (1.0 - overlap)
        * (1.0 - np.exp(-np.maximum(relative_distance, 0.0))),
        0.0,
        1.0,
    )

    quality_gate = np.clip(
        0.25 + 0.75 * endpoint_quality,
        0.25,
        1.0,
    )

    diversity_term = np.clip(
        1.0
        + 0.055 * (scarcity_term - 1.0)
        + 0.045 * bridge_signal
        - 0.030 * overlap,
        0.90,
        1.10,
    )

    long_edge_penalty = np.exp(
        -0.52 * np.maximum(relative_distance - 1.0, 0.0)
    )

    cycle_penalty = np.exp(
        -0.22 * overlap * quality_gate
    )

    scores = (
        distance_quality
        * (
            0.48
            + 0.29 * endpoint_quality
            + 0.23 * rank_quality
        )
        * regret_bonus
        * mutual_bonus
        * diversity_term
        * long_edge_penalty
        * cycle_penalty
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
