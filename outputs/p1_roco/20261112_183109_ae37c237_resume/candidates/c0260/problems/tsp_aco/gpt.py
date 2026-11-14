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

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

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
        -np.clip(1.05 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.85 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.5
            * (
                work / local_scale[:, None]
                + work / local_scale[None, :]
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.30 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.22, 1.0)

    long_edge_penalty = np.exp(
        -0.55 * np.maximum(relative_distance - 1.0, 0.0)
    )

    base_scores = (
        distance_term
        * (0.50 + 0.30 * bidirectional_rank + 0.20 * endpoint_quality)
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    # Local proximity profile used for bounded diversity and bridge signals.
    proximity = np.exp(
        -np.clip(relative_distance, 0.0, 40.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    support = np.sum(proximity, axis=1)
    support_scale = float(np.median(support[support > 0.0]))
    if not np.isfinite(support_scale) or support_scale <= 0.0:
        support_scale = 1.0

    # Nodes with weak local support receive a small exploration bonus.
    rarity = np.sqrt(
        support_scale / np.maximum(support, eps)
    )
    rarity = np.clip(rarity, 0.75, 1.75)
    rarity_pair = 0.5 * (
        rarity[:, None] + rarity[None, :]
    )

    # Cosine overlap estimates whether an edge stays inside one neighborhood.
    profile_norm = np.sqrt(
        np.sum(proximity * proximity, axis=1)
    )
    overlap = np.divide(
        proximity @ proximity.T,
        profile_norm[:, None] * profile_norm[None, :],
        out=np.zeros((n, n), dtype=float),
        where=(
            profile_norm[:, None] * profile_norm[None, :]
        ) > 0.0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    # Bridge signal is capped so long, noisy edges cannot dominate.
    bridge_signal = np.clip(
        (1.0 - overlap) * (1.0 - np.exp(-relative_distance)),
        0.0,
        1.0,
    )

    # Endpoint feasibility: preserve alternatives for both degree-2 endpoints.
    feasible_rank = np.exp(
        -ranks / max(1.0, 1.15 * np.sqrt(float(n)))
    )
    alternative_mass = np.sum(
        feasible_rank * valid,
        axis=1,
    )
    alternative_scale = float(
        np.median(alternative_mass[alternative_mass > 0.0])
    )
    if not np.isfinite(alternative_scale) or alternative_scale <= 0.0:
        alternative_scale = 1.0

    scarcity = np.clip(
        alternative_scale / np.maximum(alternative_mass, eps),
        0.5,
        1.5,
    )
    endpoint_scarcity = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )

    # Weak, bounded corrections:
    # - diversity rewards sparse endpoints and cross-neighborhood edges;
    # - potential rewards degree-feasible edges and mildly penalizes closure.
    diversity_factor = np.clip(
        1.0
        + 0.055 * (rarity_pair - 1.0)
        + 0.045 * bridge_signal
        - 0.035 * overlap,
        0.90,
        1.10,
    )

    potential_factor = np.clip(
        1.0
        + 0.045 * (endpoint_scarcity - 1.0)
        + 0.045 * bridge_signal
        - 0.040 * overlap,
        0.90,
        1.10,
    )

    scores = base_scores * diversity_factor * potential_factor
    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
