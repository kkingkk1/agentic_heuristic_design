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

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    values = d[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    global_scale = (
        float(np.median(positive))
        if positive.size
        else float(np.max(values))
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(values))
    penalty = 5.0 * max(finite_max, global_scale) + global_scale

    work = np.where(valid, d, penalty)
    np.fill_diagonal(work, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    sorted_distances = np.sort(work, axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else best

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
        work,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    distance_term = np.exp(
        -np.clip(1.10 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.88 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(rank_strength * rank_strength.T)

    endpoint_quality = np.exp(
        -np.clip(
            0.46
            * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )
    quality_gate = np.clip(
        0.48 + 0.52 * endpoint_quality,
        0.48,
        1.0,
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)
    regret_bonus = 1.0 + 0.34 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.28, 1.0)

    long_edge_penalty = np.exp(
        -0.68 * np.maximum(relative_distance - 1.0, 0.0)
    )

    affinity = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 40.0)
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    node_mass = np.sum(affinity, axis=1)
    positive_mass = node_mass[node_mass > 0.0]
    mean_mass = (
        float(np.mean(positive_mass))
        if positive_mass.size
        else 1.0
    )
    scarcity = np.sqrt(
        np.divide(mean_mass, np.maximum(node_mass, eps))
    )
    scarcity = np.clip(scarcity, 0.90, 1.16)
    scarcity_term = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )

    neighbor_count = min(
        n - 1,
        max(2, int(np.sqrt(float(n)))),
    )
    neighbor_mask = np.zeros((n, n), dtype=bool)
    neighbor_mask[
        np.arange(n)[:, None],
        order[:, :neighbor_count],
    ] = True
    neighbor_mask &= valid

    neighborhoods = neighbor_mask.astype(float)
    overlap = neighborhoods @ neighborhoods.T
    overlap /= float(max(1, neighbor_count))

    bridge_quality = np.sqrt(
        np.clip(
            affinity
            / np.maximum(
                np.max(affinity, axis=1, keepdims=True),
                eps,
            ),
            0.0,
            1.0,
        )
    )
    bridge_quality *= bridge_quality.T

    cross_region = 1.0 - np.clip(overlap, 0.0, 1.0)
    bridge_term = (
        1.0
        + 0.14
        * cross_region
        * bridge_quality
        * quality_gate
    )

    two_step = affinity @ affinity.T
    two_step_scale = np.maximum(
        np.max(two_step, axis=1, keepdims=True),
        eps,
    )
    two_step = two_step / np.sqrt(
        two_step_scale * two_step_scale.T
    )
    two_step = np.clip(two_step, 0.0, 1.0)
    long_term_term = 1.0 + 0.18 * two_step * quality_gate

    exploratory_term = (
        1.0
        + quality_gate
        * (
            0.42 * (scarcity_term - 1.0)
            + (bridge_term - 1.0)
            + (long_term_term - 1.0)
        )
    )

    scores = (
        distance_term
        * (
            0.52
            + 0.28 * bidirectional_rank
            + 0.20 * endpoint_quality
        )
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
        * exploratory_term
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
