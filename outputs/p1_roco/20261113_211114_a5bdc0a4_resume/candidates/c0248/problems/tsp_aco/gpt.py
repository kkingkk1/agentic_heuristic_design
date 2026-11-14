import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term-aware TSP edge-selection priors."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)

    # An undirected TSP edge is usable only when both directions are finite.
    finite = (
        np.isfinite(raw)
        & np.isfinite(raw.T)
        & ~diagonal
    )

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(np.isfinite(raw), raw, 0.0)
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[finite]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
        q25 = float(np.percentile(positive, 25.0))
        q75 = float(np.percentile(positive, 75.0))
    else:
        scale = 1.0
        q25 = 0.0
        q75 = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    spread = max(
        q75 - q25,
        0.20 * scale,
        np.finfo(float).eps,
    )

    maximum_observed = (
        float(np.max(observed))
        if observed.size
        else scale
    )
    missing_penalty = maximum_observed + 8.0 * scale

    distances = np.where(finite, cleaned, missing_penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, missing_penalty)

    valid_values = distances[finite]
    positive_valid = valid_values[valid_values > 0.0]

    temperature = (
        float(np.median(positive_valid))
        if positive_valid.size
        else scale
    )
    temperature = max(
        0.32 * temperature,
        0.22 * spread,
        np.finfo(float).eps,
    )

    affinity = np.exp(
        -np.minimum(distances, missing_penalty) / temperature
    )
    affinity[~finite] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if not np.isfinite(affinity_max) or affinity_max <= 0.0:
        result = np.zeros((n, n), dtype=float)
        result[finite] = 1.0
        return result

    affinity /= affinity_max

    # Soft candidate distributions describe each node's local search basin.
    valid_degree = finite.sum(axis=1).astype(float)
    valid_degree = np.maximum(valid_degree, 1.0)

    row_mass = affinity.sum(axis=1, keepdims=True)
    probabilities = affinity / np.maximum(
        row_mass,
        np.finfo(float).eps,
    )

    entropy = np.zeros(n, dtype=float)
    for i in range(n):
        degree = int(valid_degree[i])
        if degree <= 1:
            entropy[i] = 0.0
            continue

        row = probabilities[i]
        entropy[i] = float(
            -np.sum(row[row > 0.0] * np.log(row[row > 0.0]))
            / max(np.log(float(degree)), 1.0)
        )

    entropy = np.clip(np.nan_to_num(entropy), 0.0, 1.0)
    concentration = np.clip(1.0 - entropy, 0.0, 1.0)

    # Rank-aware endpoint utility, with a bounded reward for clear local winners.
    endpoint = np.zeros((n, n), dtype=float)

    for i in range(n):
        valid = np.flatnonzero(finite[i])
        if valid.size == 0:
            continue

        values = distances[i, valid]
        order = np.argsort(values, kind="stable")
        ranks = np.empty(valid.size, dtype=float)
        ranks[order] = np.arange(valid.size, dtype=float)

        rank_signal = np.exp(
            -0.95 * ranks / max(float(valid.size - 1), 1.0)
        )
        endpoint[i, valid] = (
            0.70 * affinity[i, valid]
            + 0.30 * rank_signal
        )

        if valid.size >= 2:
            best = float(values[order[0]])
            second = float(values[order[1]])
            gap = max(second - best, 0.10 * spread)
            winner_bonus = 0.10 * np.exp(
                np.clip((second - best) / gap, -4.0, 4.0)
            )
            endpoint[i, valid[order[0]]] += winner_bonus

    endpoint = np.sqrt(np.maximum(endpoint, 0.0) * np.maximum(endpoint.T, 0.0))
    endpoint[~finite] = 0.0

    endpoint_max = float(np.max(endpoint))
    if endpoint_max > 0.0 and np.isfinite(endpoint_max):
        endpoint /= endpoint_max

    # Replacement pressure estimates how costly it is to discard an edge.
    replacement = np.zeros((n, n), dtype=float)

    for i in range(n):
        valid = np.flatnonzero(finite[i])
        if valid.size == 0:
            continue

        values = distances[i, valid]
        sorted_values = np.sort(values, kind="stable")

        if sorted_values.size >= 3:
            reserve = float(np.mean(sorted_values[1:3]))
        elif sorted_values.size == 2:
            reserve = float(sorted_values[1])
        else:
            reserve = float(sorted_values[0])

        for position, j in enumerate(valid):
            alternatives = np.delete(values, position)
            fallback = (
                float(np.min(alternatives))
                if alternatives.size
                else reserve
            )
            pressure = max(fallback - distances[i, j], 0.0)
            replacement[i, j] = 1.0 - np.exp(
                -pressure / max(0.70 * scale, np.finfo(float).eps)
            )

    replacement = np.sqrt(
        np.maximum(replacement, 0.0)
        * np.maximum(replacement.T, 0.0)
    )
    replacement[~finite] = 0.0

    # Two-hop continuation captures whether an edge participates in
    # compatible future transitions rather than only being locally short.
    transition = endpoint.copy()
    transition_mass = transition.sum(axis=1, keepdims=True)
    transition /= np.maximum(
        transition_mass,
        np.finfo(float).eps,
    )

    two_hop = transition @ transition
    two_hop = 0.5 * (two_hop + two_hop.T)
    two_hop[~finite] = 0.0
    np.fill_diagonal(two_hop, 0.0)

    two_hop_max = float(np.max(two_hop))
    if two_hop_max > 0.0 and np.isfinite(two_hop_max):
        two_hop /= two_hop_max

    # Low-overlap edges connect structurally different neighborhoods.
    # This is a bounded bridge/diversity signal.
    neighborhood_overlap = probabilities @ probabilities.T
    neighborhood_norm = np.sqrt(
        np.maximum(np.diag(neighborhood_overlap), 0.0)
    )
    denominator = (
        neighborhood_norm[:, None] * neighborhood_norm[None, :]
    )

    similarity = neighborhood_overlap / np.maximum(
        denominator,
        np.finfo(float).eps,
    )
    similarity = np.clip(np.nan_to_num(similarity), 0.0, 1.0)

    bridge_value = 1.0 - similarity
    bridge_value = 0.5 * (bridge_value + bridge_value.T)
    bridge_value[~finite] = 0.0
    np.fill_diagonal(bridge_value, 0.0)

    # Candidate scarcity is used only as a gate. It does not independently
    # dominate distance, which protects short edges in concentrated regions.
    scarcity = np.sqrt(
        concentration[:, None] * concentration[None, :]
    )
    scarcity *= np.sqrt(
        valid_degree[:, None] * valid_degree[None, :]
    ) / max(float(np.max(valid_degree)), 1.0)
    scarcity[~finite] = 0.0

    # Adaptive long-term weights: concentrated nodes receive stronger
    # continuation and bridge signals, while diffuse instances remain
    # primarily distance-driven.
    concentration_level = float(np.mean(concentration))
    continuation_weight = 0.10 + 0.10 * concentration_level
    bridge_weight = 0.07 + 0.12 * concentration_level
    replacement_weight = 0.13 + 0.04 * concentration_level

    long_term = (
        continuation_weight * two_hop
        + replacement_weight * replacement
        + bridge_weight * bridge_value * (0.35 + 0.65 * scarcity)
    )
    long_term[~finite] = 0.0
    long_term = 0.5 * (long_term + long_term.T)

    # Main score keeps robust distance affinity dominant.
    score = (
        0.60 * affinity
        + 0.18 * endpoint
        + 0.22 * long_term
    )

    # Add a small diversity bonus, reduced on already excellent short edges.
    short_edge_protection = 0.30 + 0.70 * (1.0 - affinity)
    score += 0.06 * bridge_value * scarcity * short_edge_protection

    score[~finite] = 0.0
    score = 0.5 * (score + score.T)

    # Symmetric diagonal balancing reduces node monopolies without imposing
    # a hard uniform-degree constraint.
    balanced = score.copy()

    for _ in range(28):
        row_sum = balanced.sum(axis=1)
        scale_vector = 1.0 / np.sqrt(
            np.maximum(row_sum, np.finfo(float).eps)
        )
        balanced *= scale_vector[:, None] * scale_vector[None, :]
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~finite] = 0.0

        # Damping avoids over-correcting sparse or highly heterogeneous rows.
        balanced = 0.78 * balanced + 0.22 * score

    balanced[~finite] = 0.0
    balanced = 0.5 * (balanced + balanced.T)

    # Preserve the strongest distance signal after balancing.
    balance_weight = 0.22 + 0.16 * concentration_level
    result = (
        (1.0 - balance_weight) * score
        + balance_weight * balanced
    )

    # A floor-like protected component prevents balancing from erasing
    # genuinely short edges.
    result += 0.035 * affinity
    result[~finite] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
