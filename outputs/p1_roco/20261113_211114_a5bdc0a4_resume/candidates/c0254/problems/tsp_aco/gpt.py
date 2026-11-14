import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & np.isfinite(raw.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(np.isfinite(raw), raw, 0.0)
    cleaned = np.maximum(cleaned, 0.0)
    distances = 0.5 * (cleaned + cleaned.T)

    observed = distances[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        nonzero = observed[observed != 0.0]
        scale = float(np.median(np.abs(nonzero))) if nonzero.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    upper = float(np.percentile(observed, 95.0)) if observed.size else scale
    cap = max(upper + 4.0 * scale, scale)

    distances = np.minimum(distances, cap)
    distances[~valid] = cap
    np.fill_diagonal(distances, cap)

    temperature = max(0.72 * scale, np.finfo(float).eps)
    affinity = np.exp(-distances / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if not np.isfinite(affinity_max) or affinity_max <= 0.0:
        affinity_max = 1.0
    affinity = affinity / affinity_max

    local_rank = np.zeros((n, n), dtype=float)
    endpoint_bonus = np.zeros((n, n), dtype=float)
    node_entropy = np.zeros(n, dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[
            np.argsort(distances[i, candidates], kind="stable")
        ]

        ranks = np.arange(order.size, dtype=float)
        denominator = max(float(order.size - 1), 1.0)
        local_rank[i, order] = 1.0 - ranks / denominator

        row_affinity = affinity[i, candidates]
        mass = float(row_affinity.sum())
        if mass > 0.0 and candidates.size > 1:
            probabilities = row_affinity / mass
            node_entropy[i] = float(
                -np.sum(probabilities * np.log(probabilities + 1e-12))
                / np.log(float(candidates.size))
            )

        best = distances[i, order[0]]
        second = distances[i, order[1]] if order.size > 1 else best
        gap_scale = max(
            0.25 * scale,
            second - best,
            np.finfo(float).eps,
        )

        endpoint_bonus[i, order[0]] = 1.0 - np.exp(
            -max(second - best, 0.0) / gap_scale
        )

    local_rank = np.sqrt(local_rank * local_rank.T)
    endpoint_bonus = np.sqrt(endpoint_bonus * endpoint_bonus.T)

    replacement_pressure = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        values = distances[i, candidates]
        for position, j in enumerate(candidates):
            alternatives = np.delete(values, position)
            if alternatives.size:
                fallback = float(np.min(alternatives))
            else:
                fallback = float(values[position])

            pressure = max(fallback - distances[i, j], 0.0)
            replacement_pressure[i, j] = 1.0 - np.exp(
                -pressure / max(0.75 * scale, np.finfo(float).eps)
            )

    replacement_pressure = np.sqrt(
        replacement_pressure * replacement_pressure.T
    )
    replacement_pressure[~valid] = 0.0

    transition = (
        0.72 * affinity
        + 0.20 * local_rank
        + 0.08 * endpoint_bonus
    )
    transition[~valid] = 0.0

    row_sum = transition.sum(axis=1, keepdims=True)
    transition = transition / np.maximum(row_sum, np.finfo(float).eps)

    two_hop = transition @ transition
    two_hop = 0.5 * (two_hop + two_hop.T)
    two_hop[~valid] = 0.0
    np.fill_diagonal(two_hop, 0.0)

    two_hop_max = float(np.max(two_hop))
    if np.isfinite(two_hop_max) and two_hop_max > 0.0:
        two_hop /= two_hop_max

    scarcity = np.clip(1.0 - node_entropy, 0.0, 1.0)
    degree = np.maximum(valid.sum(axis=1), 1)
    scarcity *= np.sqrt(degree / max(float(np.max(degree)), 1.0))

    edge_mass = transition.sum(axis=0) + transition.sum(axis=1)
    inverse_mass = 1.0 / np.sqrt(
        np.maximum(edge_mass, np.finfo(float).eps)
    )
    inverse_mass /= max(float(np.max(inverse_mass)), np.finfo(float).eps)

    gated_scarcity = np.sqrt(
        (scarcity * inverse_mass)[:, None]
        * (scarcity * inverse_mass)[None, :]
    )
    gated_scarcity[~valid] = 0.0

    potential = (
        0.50 * two_hop
        + 0.35 * replacement_pressure
        + 0.15 * gated_scarcity
    )
    potential[~valid] = 0.0
    potential = 0.5 * (potential + potential.T)

    score = (
        0.68 * affinity
        + 0.16 * local_rank
        + 0.06 * endpoint_bonus
        + 0.07 * replacement_pressure
        + 0.03 * two_hop
    )
    score *= 0.86 + 0.14 * potential
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    diversity = 0.06 * np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    row_sum = score.sum(axis=1, keepdims=True)
    normalized = score / np.maximum(row_sum, np.finfo(float).eps)
    score = (1.0 - diversity) * score + diversity * normalized
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(12):
        row_sum = balanced.sum(axis=1, keepdims=True)
        normalized = balanced / np.maximum(
            row_sum, np.finfo(float).eps
        )
        balanced = 0.82 * balanced + 0.18 * normalized
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    concentration = float(np.mean(1.0 - node_entropy))
    balance_weight = float(np.clip(
        0.12 + 0.08 * concentration,
        0.12,
        0.20,
    ))

    result = (
        (1.0 - balance_weight) * score
        + balance_weight * balanced
    )
    result = 0.5 * (result + result.T)
    result[~valid] = 0.0
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
