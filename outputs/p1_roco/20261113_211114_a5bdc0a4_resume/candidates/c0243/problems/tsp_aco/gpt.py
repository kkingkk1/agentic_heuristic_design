import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diversified edge-selection priors for stochastic TSP sampling."""
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
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_max = float(np.max(observed)) if observed.size else scale
    penalty = finite_max + 8.0 * scale

    distances = np.where(finite, cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    valid_values = distances[~diagonal]
    valid_values = valid_values[valid_values > 0.0]
    temperature = float(np.median(valid_values)) if valid_values.size else scale
    temperature = max(0.26 * temperature, np.finfo(float).eps)

    affinity = np.exp(-np.minimum(distances, penalty) / temperature)
    affinity[~finite] = 0.0
    np.fill_diagonal(affinity, 0.0)

    if not np.any(affinity > 0.0):
        result = np.ones((n, n), dtype=float)
        result[~finite] = 0.0
        np.fill_diagonal(result, 0.0)
        return result

    endpoint_gap = np.zeros((n, n), dtype=float)

    for i in range(n):
        row = distances[i].copy()
        row[i] = penalty
        order = np.argsort(row, kind="stable")

        best = row[order[0]]
        second = row[order[1]] if n > 2 else best
        local_scale = max(second - best, 0.16 * scale)
        local_scale = max(local_scale, np.finfo(float).eps)

        for j in range(n):
            if i == j or not finite[i, j]:
                continue

            if j == order[0]:
                gap = second - distances[i, j]
            else:
                gap = best - distances[i, j]

            endpoint_gap[i, j] = np.exp(
                np.clip(gap / local_scale, -22.0, 22.0)
            )

    endpoint_gap = np.sqrt(endpoint_gap * endpoint_gap.T)
    endpoint_gap[~finite] = 0.0

    best_bypass = np.full((n, n), penalty, dtype=float)

    for k in range(n):
        candidate = distances[:, k, None] + distances[None, k, :]
        best_bypass = np.minimum(best_bypass, candidate)

    np.fill_diagonal(best_bypass, penalty)

    bypass_advantage = np.maximum(best_bypass - distances, 0.0)
    bypass_scale = max(0.72 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(-bypass_advantage / bypass_scale)
    irreplaceability[~finite] = 0.0

    max_affinity = max(float(np.max(affinity)), np.finfo(float).eps)
    base = affinity / max_affinity

    local_score = (
        0.50 * base
        + 0.20 * endpoint_gap
        + 0.30 * irreplaceability
    )
    local_score[~finite] = 0.0
    local_score = 0.5 * (local_score + local_score.T)

    row_total = local_score.sum(axis=1, keepdims=True)
    row_total = np.maximum(row_total, np.finfo(float).eps)
    transition = local_score / row_total

    two_step = transition @ transition
    two_step = 0.5 * (two_step + two_step.T)
    two_step[~finite] = 0.0
    np.fill_diagonal(two_step, 0.0)

    continuation = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if not finite[i, j]:
                continue

            alternatives_i = np.delete(distances[i], [i, j])
            alternatives_j = np.delete(distances[j], [i, j])

            if alternatives_i.size:
                reserve_i = float(np.min(alternatives_i))
            else:
                reserve_i = distances[i, j]

            if alternatives_j.size:
                reserve_j = float(np.min(alternatives_j))
            else:
                reserve_j = distances[i, j]

            reserve = reserve_i + reserve_j
            direct = 2.0 * distances[i, j]
            value = np.exp(
                np.clip((reserve - direct) / max(scale, np.finfo(float).eps),
                        -22.0, 22.0)
            )
            continuation[i, j] = value
            continuation[j, i] = value

    continuation[~finite] = 0.0
    if np.any(continuation > 0.0):
        continuation /= max(
            float(np.max(continuation)),
            np.finfo(float).eps,
        )

    edge_mass = local_score.sum(axis=0) + local_score.sum(axis=1)
    edge_mass = np.maximum(edge_mass, np.finfo(float).eps)
    rarity = 1.0 / np.sqrt(edge_mass)
    rarity /= max(float(np.max(rarity)), np.finfo(float).eps)
    rarity_matrix = np.sqrt(rarity[:, None] * rarity[None, :])
    rarity_matrix[~finite] = 0.0

    global_potential = (
        0.55 * two_step
        + 0.25 * continuation
        + 0.20 * rarity_matrix
    )
    global_potential[~finite] = 0.0
    global_potential = 0.5 * (global_potential + global_potential.T)

    score = local_score * (0.72 + 0.28 * global_potential)
    score[~finite] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(40):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~finite] = 0.0

    result = 0.60 * score + 0.40 * balanced
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
