import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & np.isfinite(raw.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.where(np.isfinite(raw), raw, 0.0)
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    values = distances[valid]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_values = values[np.isfinite(values)]
    upper = (
        float(np.percentile(finite_values, 95.0))
        if finite_values.size
        else scale
    )
    cap = max(upper + 3.0 * scale, scale)

    distances = np.minimum(distances, cap)
    distances[~valid] = cap
    np.fill_diagonal(distances, cap)

    local_scale = np.full(n, scale, dtype=float)
    local_second = np.full(n, cap, dtype=float)
    ranks = np.zeros((n, n), dtype=float)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[
            np.argsort(distances[i, candidates], kind="stable")
        ]
        nearest[i] = int(order[0])

        best = distances[i, order[0]]
        if order.size > 1:
            local_second[i] = distances[i, order[1]]

        deviations = distances[i, order] - best
        if deviations.size > 1:
            local_scale[i] = max(
                float(np.median(deviations[1:])),
                0.20 * scale,
                np.finfo(float).eps,
            )
        else:
            local_scale[i] = max(
                0.20 * scale,
                np.finfo(float).eps,
            )

        denominator = max(float(order.size - 1), 1.0)
        ranks[i, order] = 1.0 - np.arange(order.size) / denominator

    temperature = np.sqrt(local_scale[:, None] * local_scale[None, :])
    temperature = np.maximum(temperature, 0.35 * scale)

    affinity = np.exp(-distances / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if not np.isfinite(affinity_max) or affinity_max <= 0.0:
        affinity_max = 1.0
    affinity /= affinity_max

    rank_score = np.sqrt(np.maximum(ranks, 0.0) * np.maximum(ranks.T, 0.0))

    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_bonus[i, j] = 1.0

    mutual_bonus = 0.5 * (mutual_bonus + mutual_bonus.T)

    gap = np.maximum(local_second[:, None] - distances, 0.0)
    gap_scale = np.maximum(local_scale[:, None], np.finfo(float).eps)
    endpoint_gap = 1.0 - np.exp(-gap / gap_scale)
    endpoint_gap[~valid] = 0.0
    endpoint_gap = np.sqrt(
        np.maximum(endpoint_gap, 0.0)
        * np.maximum(endpoint_gap.T, 0.0)
    )

    base_score = (
        0.78 * affinity
        + 0.12 * rank_score
        + 0.06 * endpoint_gap
        + 0.04 * mutual_bonus
    )
    base_score[~valid] = 0.0
    np.fill_diagonal(base_score, 0.0)
    base_score = 0.5 * (base_score + base_score.T)

    # Soft local candidate distributions for adaptive exploration gating.
    row_mass = affinity.sum(axis=1, keepdims=True)
    probabilities = affinity / np.maximum(
        row_mass,
        np.finfo(float).eps,
    )

    degree = valid.sum(axis=1).astype(float)
    entropy = np.zeros(n, dtype=float)

    for i in range(n):
        if degree[i] <= 1.0:
            continue
        row = probabilities[i]
        positive_row = row[row > 0.0]
        entropy[i] = float(
            -np.sum(positive_row * np.log(positive_row))
            / max(np.log(degree[i]), 1.0)
        )

    entropy = np.clip(np.nan_to_num(entropy), 0.0, 1.0)
    concentration = 1.0 - entropy
    concentration_pair = np.sqrt(
        concentration[:, None] * concentration[None, :]
    )
    concentration_pair[~valid] = 0.0

    # Weak two-hop continuation signal.
    transition = base_score.copy()
    transition_mass = transition.sum(axis=1, keepdims=True)
    transition /= np.maximum(
        transition_mass,
        np.finfo(float).eps,
    )

    continuation = transition @ transition
    continuation = 0.5 * (continuation + continuation.T)
    continuation[~valid] = 0.0
    np.fill_diagonal(continuation, 0.0)

    continuation_max = float(np.max(continuation))
    if np.isfinite(continuation_max) and continuation_max > 0.0:
        continuation /= continuation_max

    # Low neighborhood overlap indicates a possible bridge between regions.
    overlap = probabilities @ probabilities.T
    norm = np.sqrt(
        np.maximum(np.diag(overlap), 0.0)
    )
    similarity = overlap / np.maximum(
        norm[:, None] * norm[None, :],
        np.finfo(float).eps,
    )
    similarity = np.clip(np.nan_to_num(similarity), 0.0, 1.0)

    bridge = 1.0 - similarity
    bridge = 0.5 * (bridge + bridge.T)
    bridge[~valid] = 0.0
    np.fill_diagonal(bridge, 0.0)

    concentration_level = float(np.mean(concentration))
    continuation_weight = 0.025 + 0.035 * concentration_level
    bridge_weight = 0.015 + 0.025 * concentration_level

    score = (
        base_score
        + continuation_weight * continuation * concentration_pair
        + bridge_weight * bridge * concentration_pair
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    # Mild normalization prevents node concentration without global
    # balancing that could erase genuinely short edges.
    row_sum = score.sum(axis=1, keepdims=True)
    row_normalized = score / np.maximum(
        row_sum,
        np.finfo(float).eps,
    )
    row_normalized = 0.5 * (row_normalized + row_normalized.T)

    result = 0.90 * score + 0.10 * row_normalized
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
