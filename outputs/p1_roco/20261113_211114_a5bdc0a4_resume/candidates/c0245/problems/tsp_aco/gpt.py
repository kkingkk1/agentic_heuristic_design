import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return diversified, long-term-aware edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & ~diagonal

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

    spread = max(q75 - q25, 0.20 * scale, np.finfo(float).eps)
    finite_max = float(np.max(observed)) if observed.size else scale
    missing_penalty = finite_max + 6.0 * scale

    distances = np.where(finite, cleaned, missing_penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, missing_penalty)

    positive_distances = distances[~diagonal]
    positive_distances = positive_distances[positive_distances > 0.0]
    temperature = float(np.median(positive_distances)) if positive_distances.size else scale
    temperature = max(0.34 * temperature, 0.20 * spread, np.finfo(float).eps)

    affinity = np.exp(
        -np.minimum(distances, missing_penalty) / temperature
    )
    affinity[~finite] = 0.0
    np.fill_diagonal(affinity, 0.0)

    if not np.any(affinity > 0.0):
        result = np.ones((n, n), dtype=float)
        result[~finite] = 0.0
        np.fill_diagonal(result, 0.0)
        return result

    affinity /= max(float(np.max(affinity)), np.finfo(float).eps)

    # Rank-aware endpoint preference with bounded advantage for locally promising edges.
    endpoint_score = np.zeros((n, n), dtype=float)
    node_entropy = np.zeros(n, dtype=float)
    node_degree = np.maximum(finite.sum(axis=1), 1)

    for i in range(n):
        valid = np.flatnonzero(finite[i])
        if valid.size == 0:
            continue

        values = distances[i, valid]
        order = valid[np.argsort(values, kind="stable")]
        ranks = np.empty(valid.size, dtype=float)
        ranks[np.argsort(np.argsort(values, kind="stable"), kind="stable")] = (
            np.arange(valid.size, dtype=float)
        )

        row_affinity = affinity[i, valid]
        row_mass = float(row_affinity.sum())
        if row_mass > 0.0:
            probabilities = row_affinity / row_mass
            node_entropy[i] = float(
                -np.sum(probabilities * np.log(probabilities + 1e-12))
                / max(np.log(float(valid.size)), 1.0)
            )

        rank_decay = np.exp(-0.85 * ranks / max(float(valid.size - 1), 1.0))
        endpoint_score[i, valid] = 0.68 * row_affinity + 0.32 * rank_decay

        if order.size >= 2:
            best = distances[i, order[0]]
            second = distances[i, order[1]]
            local_gap = max(second - best, 0.12 * spread)
            endpoint_score[i, order[0]] += 0.12 * np.exp(
                np.clip((second - best) / local_gap, -5.0, 5.0)
            )

    endpoint_score = np.sqrt(endpoint_score * endpoint_score.T)
    endpoint_score[~finite] = 0.0
    endpoint_score /= max(float(np.max(endpoint_score)), np.finfo(float).eps)

    # Estimate the cost of losing an edge at either endpoint.
    replacement_pressure = np.zeros((n, n), dtype=float)

    for i in range(n):
        valid = np.flatnonzero(finite[i])
        if valid.size == 0:
            continue

        sorted_values = np.sort(distances[i, valid], kind="stable")
        if sorted_values.size >= 3:
            reserve = sorted_values[1:3].mean()
        elif sorted_values.size == 2:
            reserve = sorted_values[1]
        else:
            reserve = sorted_values[0]

        for j in valid:
            alternatives = np.delete(distances[i, valid], np.where(valid == j)[0])
            if alternatives.size:
                fallback = float(np.min(alternatives))
            else:
                fallback = reserve

            pressure = fallback - distances[i, j]
            replacement_pressure[i, j] = 1.0 - np.exp(
                -max(pressure, 0.0) / max(0.75 * scale, np.finfo(float).eps)
            )

    replacement_pressure = np.sqrt(
        replacement_pressure * replacement_pressure.T
    )
    replacement_pressure[~finite] = 0.0

    # Two-hop potential captures whether an edge leaves useful continuations.
    local_transition = endpoint_score.copy()
    row_sum = local_transition.sum(axis=1, keepdims=True)
    local_transition /= np.maximum(row_sum, np.finfo(float).eps)

    two_hop = local_transition @ local_transition
    two_hop = 0.5 * (two_hop + two_hop.T)
    two_hop[~finite] = 0.0
    np.fill_diagonal(two_hop, 0.0)
    if np.any(two_hop > 0.0):
        two_hop /= max(float(np.max(two_hop)), np.finfo(float).eps)

    # Entropy-gated scarcity: only low-entropy nodes receive a modest diversity bonus.
    scarcity_strength = np.clip(1.0 - node_entropy, 0.0, 1.0)
    scarcity_strength *= np.sqrt(node_degree / max(float(np.max(node_degree)), 1.0))

    edge_mass = endpoint_score.sum(axis=0) + endpoint_score.sum(axis=1)
    edge_mass = np.maximum(edge_mass, np.finfo(float).eps)
    inverse_mass = 1.0 / np.sqrt(edge_mass)
    inverse_mass /= max(float(np.max(inverse_mass)), np.finfo(float).eps)

    gated_scarcity = np.sqrt(
        (scarcity_strength * inverse_mass)
        [:, None]
        * (scarcity_strength * inverse_mass)[None, :]
    )
    gated_scarcity[~finite] = 0.0

    # Conservative long-term potential. Distance affinity remains dominant.
    potential = (
        0.52 * two_hop
        + 0.30 * replacement_pressure
        + 0.18 * gated_scarcity
    )
    potential[~finite] = 0.0
    potential = 0.5 * (potential + potential.T)

    score = (
        0.56 * affinity
        + 0.24 * endpoint_score
        + 0.20 * replacement_pressure
    )
    score *= 0.78 + 0.22 * potential
    score[~finite] = 0.0
    score = 0.5 * (score + score.T)

    # Mild entropy-aware diversification before matrix balancing.
    row_mass = score.sum(axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, np.finfo(float).eps)
    normalized = score / row_mass

    diversity_mix = 0.08 * np.sqrt(
        np.maximum(scarcity_strength[:, None] * scarcity_strength[None, :], 0.0)
    )
    score = (1.0 - diversity_mix) * score + diversity_mix * normalized
    score[~finite] = 0.0
    score = 0.5 * (score + score.T)

    # Damped symmetric balancing prevents a few nodes from monopolizing probability mass.
    balanced = score.copy()
    for _ in range(36):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        scaled = balanced / row_sum
        balanced = 0.72 * balanced + 0.28 * scaled
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~finite] = 0.0

    # More diverse blending for concentrated candidate distributions.
    concentration = float(np.mean(1.0 - node_entropy))
    balanced_weight = float(np.clip(0.30 + 0.14 * concentration, 0.30, 0.44))

    result = (
        (1.0 - balanced_weight) * score
        + balanced_weight * balanced
    )
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
