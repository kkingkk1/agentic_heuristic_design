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
    penalty = max(finite_max, global_scale) * 5.0 + global_scale

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
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.85 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )

    endpoint_quality = np.exp(
        -np.clip(
            0.45
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

    regret_bonus = 1.0 + 0.40 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    indices = np.arange(n)
    mutual_nearest = (
        (nearest[:, None] == indices[None, :])
        & (nearest[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.38, 1.0)

    long_edge_penalty = np.exp(
        -0.72 * np.maximum(relative_distance - 1.0, 0.0)
    )

    base_scores = (
        distance_term
        * (0.52 + 0.28 * bidirectional_rank + 0.20 * endpoint_quality)
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    # Local similarity describes each node's accessible neighborhood.
    similarity = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 30.0)
    )
    similarity[~valid] = 0.0
    np.fill_diagonal(similarity, 0.0)

    neighborhood_mass = np.sum(similarity, axis=1)
    mass_scale = max(float(np.median(neighborhood_mass)), eps)

    # Static ambiguity is a proxy for where extra exploration may be useful.
    strongest = np.max(similarity, axis=1)
    ambiguity = 1.0 - np.divide(
        strongest,
        np.maximum(neighborhood_mass, eps),
    )
    ambiguity = np.clip(ambiguity, 0.0, 1.0)

    # Global-diversity component:
    # reward sparse neighborhoods, neighborhood imbalance, and ambiguity.
    rarity = 1.0 / (1.0 + neighborhood_mass / mass_scale)
    rarity /= max(float(np.max(rarity)), eps)

    bridge_difference = np.divide(
        np.abs(
            neighborhood_mass[:, None]
            - neighborhood_mass[None, :]
        ),
        neighborhood_mass[:, None]
        + neighborhood_mass[None, :]
        + eps,
    )

    edge_ambiguity = 0.5 * (
        ambiguity[:, None] + ambiguity[None, :]
    )
    edge_rarity = 0.5 * (
        rarity[:, None] + rarity[None, :]
    )

    diversity_raw = (
        0.45 * edge_rarity
        + 0.35 * bridge_difference
        + 0.20 * edge_ambiguity
    )

    proximity_gate = np.exp(
        -0.55 * np.maximum(relative_distance - 1.0, 0.0)
    )
    diversity_raw *= proximity_gate
    diversity_raw[~valid] = 0.0

    diversity_scale = max(float(np.max(diversity_raw)), eps)
    diversity_component = np.clip(
        diversity_raw / diversity_scale,
        0.0,
        1.0,
    )

    # Long-term-potential component:
    # estimate whether selecting an edge leaves each endpoint with
    # alternative usable neighbors, then weakly discourage dense
    # same-neighborhood links that may close premature subtours.
    residual_similarity = similarity.copy()
    residual_similarity[valid] *= (
        1.0 - np.eye(n, dtype=float)[valid]
    )

    residual_mass = np.sum(residual_similarity, axis=1)
    residual_fraction = np.divide(
        residual_mass,
        np.maximum(neighborhood_mass, eps),
    )
    residual_fraction = np.clip(residual_fraction, 0.0, 1.0)

    endpoint_feasibility = np.sqrt(
        residual_fraction[:, None] * residual_fraction[None, :]
    )

    row_norm = np.sqrt(
        np.sum(similarity * similarity, axis=1)
    )
    overlap = np.divide(
        similarity @ similarity.T,
        row_norm[:, None] * row_norm[None, :] + eps,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    potential_raw = (
        0.68 * endpoint_feasibility
        + 0.32 * (1.0 - overlap)
    )
    potential_raw *= proximity_gate
    potential_raw[~valid] = 0.0

    potential_scale = max(float(np.max(potential_raw)), eps)
    potential_component = np.clip(
        potential_raw / potential_scale,
        0.0,
        1.0,
    )

    scores = base_scores
    scores *= 1.0 + 0.10 * diversity_component
    scores *= 1.0 + 0.12 * potential_component

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
