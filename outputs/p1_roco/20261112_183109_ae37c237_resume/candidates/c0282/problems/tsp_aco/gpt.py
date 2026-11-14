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

    finite_max = float(np.max(finite_values))
    penalty = max(finite_max, global_scale) * 4.0 + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_work = np.take_along_axis(work, order, axis=1)
    best = sorted_work[:, 0]
    second = sorted_work[:, 1] if n > 2 else best.copy()

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
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (geometric_scale > 0.0),
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

    distance_quality = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    endpoint_quality = np.exp(
        -np.clip(0.48 * (row_ratio + col_ratio), 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.90 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(
        np.nan_to_num(regret, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    regret_bonus = 1.0 + 0.22 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
        & valid
    )
    mutual_bonus = np.where(mutual_nearest, 1.18, 1.0)

    k = min(4, n - 1)
    neighbor_mask = valid & (ranks < float(k))
    neighbor_mask_float = neighbor_mask.astype(float)

    candidate_count = neighbor_mask_float.sum(axis=1)
    scarcity = 1.0 / np.sqrt(
        np.maximum(candidate_count, 1.0)
    )
    scarcity_scale = max(float(np.max(scarcity)), eps)
    scarcity = scarcity / scarcity_scale
    scarcity_bonus = 1.0 + 0.10 * (
        scarcity[:, None] + scarcity[None, :]
    )

    local_affinity = np.exp(
        -np.clip(
            d / np.maximum(local_scale[:, None], eps),
            0.0,
            40.0,
        )
    )
    local_affinity[~valid] = 0.0
    np.fill_diagonal(local_affinity, 0.0)

    candidate_affinity = local_affinity * neighbor_mask_float

    support = candidate_affinity.sum(axis=1)
    support_scale = float(np.median(support[support > 0.0]))
    if not np.isfinite(support_scale) or support_scale <= 0.0:
        support_scale = 1.0

    endpoint_rarity = np.clip(
        np.sqrt(
            support_scale / np.maximum(support, eps)
        ),
        0.75,
        1.50,
    )

    profile_norm = np.sqrt(
        np.sum(candidate_affinity * candidate_affinity, axis=1)
    )
    profile_overlap = np.divide(
        candidate_affinity @ candidate_affinity.T,
        profile_norm[:, None] * profile_norm[None, :],
        out=np.zeros((n, n), dtype=float),
        where=(
            profile_norm[:, None] * profile_norm[None, :]
        ) > 0.0,
    )
    profile_overlap = np.clip(profile_overlap, 0.0, 1.0)

    alternative = candidate_affinity.copy()
    nearest_valid = valid & (
        np.arange(n)[None, :] == nearest[:, None]
    )
    alternative[nearest_valid] = 0.0
    alternative_support = np.max(alternative, axis=1)
    alternative_support = np.clip(alternative_support, 0.0, 1.0)

    moderate_gate = np.exp(
        -0.85 * np.maximum(relative_distance - 1.05, 0.0)
    )

    bridge_signal = (
        moderate_gate
        * (1.0 - profile_overlap)
        * np.sqrt(
            alternative_support[:, None]
            * alternative_support[None, :]
        )
    )
    bridge_signal = np.clip(bridge_signal, 0.0, 1.0)

    closure_signal = np.clip(
        profile_overlap
        * np.exp(-0.65 * relative_distance),
        0.0,
        1.0,
    )

    local_adjustment = np.clip(
        1.0
        + 0.045 * (
            endpoint_rarity[:, None]
            + endpoint_rarity[None, :]
            - 2.0
        )
        + 0.055 * bridge_signal
        - 0.035 * closure_signal,
        0.90,
        1.10,
    )

    scores = (
        distance_quality
        * (
            0.52
            + 0.28 * bidirectional_rank
            + 0.20 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * scarcity_bonus
        * local_adjustment
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
