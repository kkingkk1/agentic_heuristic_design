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
    finite = np.isfinite(raw) & np.isfinite(raw.T) & ~diagonal

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

    q25 = float(np.percentile(positive, 25.0)) if positive.size else 0.0
    q75 = float(np.percentile(positive, 75.0)) if positive.size else scale
    spread = max(q75 - q25, 0.2 * scale, np.finfo(float).eps)

    maximum = float(np.max(observed)) if observed.size else scale
    missing_penalty = maximum + 8.0 * scale

    distances = np.where(finite, cleaned, missing_penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, missing_penalty)

    valid_degree = finite.sum(axis=1).astype(float)
    degree_scale = max(float(np.max(valid_degree)), 1.0)

    temperature = max(
        0.35 * scale,
        0.25 * spread,
        np.finfo(float).eps,
    )

    affinity = np.exp(-np.minimum(distances, missing_penalty) / temperature)
    affinity[~finite] = 0.0
    np.fill_diagonal(affinity, 0.0)

    affinity_max = float(np.max(affinity))
    if not np.isfinite(affinity_max) or affinity_max <= 0.0:
        result = np.zeros((n, n), dtype=float)
        result[finite] = 1.0
        return result

    affinity /= affinity_max

    # Soft local candidate distributions.
    row_mass = affinity.sum(axis=1, keepdims=True)
    probabilities = affinity / np.maximum(row_mass, np.finfo(float).eps)

    entropy = np.zeros(n, dtype=float)
    for i in range(n):
        degree = int(valid_degree[i])
        if degree <= 1:
            continue
        row = probabilities[i]
        positive_row = row[row > 0.0]
        entropy[i] = float(
            -np.sum(positive_row * np.log(positive_row))
            / max(np.log(float(degree)), 1.0)
        )

    entropy = np.clip(np.nan_to_num(entropy), 0.0, 1.0)
    concentration = 1.0 - entropy

    # Rank signal and endpoint support.
    rank_score = np.zeros((n, n), dtype=float)
    regret = np.zeros((n, n), dtype=float)

    for i in range(n):
        valid = np.flatnonzero(finite[i])
        if valid.size == 0:
            continue

        values = distances[i, valid]
        order = np.argsort(values, kind="stable")
        ranks = np.empty(valid.size, dtype=float)
        ranks[order] = np.arange(valid.size, dtype=float)

        rank_score[i, valid] = np.exp(
            -1.05 * ranks / max(float(valid.size - 1), 1.0)
        )

        best = float(values[order[0]])
        if valid.size >= 2:
            second = float(values[order[1]])
        else:
            second = best + scale

        # Regret is high when rejecting this edge creates a costly fallback.
        for position, j in enumerate(valid):
            alternatives = np.delete(values, position)
            fallback = float(np.min(alternatives)) if alternatives.size else second
            loss = max(fallback - distances[i, j], 0.0)
            regret[i, j] = 1.0 - np.exp(
                -loss / max(0.75 * scale, np.finfo(float).eps)
            )

        # Slightly emphasize clear local winners without making them absolute.
        winner_gap = max(second - best, 0.1 * spread)
        rank_score[i, valid[order[0]]] += 0.12 * np.exp(
            -max(second - best, 0.0) / winner_gap
        )

    endpoint = np.sqrt(
        np.maximum(rank_score, 0.0) * np.maximum(rank_score.T, 0.0)
    )
    regret = np.sqrt(
        np.maximum(regret, 0.0) * np.maximum(regret.T, 0.0)
    )

    endpoint[~finite] = 0.0
    regret[~finite] = 0.0

    for component in (endpoint, regret):
        maximum_component = float(np.max(component))
        if maximum_component > 0.0 and np.isfinite(maximum_component):
            component /= maximum_component

    # Sparse candidate backbone: only strong local options can propagate.
    candidate = affinity * (
        0.55 * endpoint
        + 0.45 * rank_score / max(float(np.max(rank_score)), 1.0)
    )
    candidate[~finite] = 0.0

    row_max = np.max(candidate, axis=1, keepdims=True)
    threshold = 0.22 * row_max
    backbone = np.where(candidate >= threshold, candidate, 0.0)
    backbone[~finite] = 0.0

    backbone_mass = backbone.sum(axis=1, keepdims=True)
    transition = backbone / np.maximum(
        backbone_mass,
        np.finfo(float).eps,
    )

    # Two-hop continuation potential.
    continuation = transition @ transition
    continuation = 0.5 * (continuation + continuation.T)
    continuation[~finite] = 0.0
    np.fill_diagonal(continuation, 0.0)

    continuation_max = float(np.max(continuation))
    if continuation_max > 0.0 and np.isfinite(continuation_max):
        continuation /= continuation_max

    # Neighborhood dissimilarity estimates bridge value.
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
    bridge[~finite] = 0.0
    np.fill_diagonal(bridge, 0.0)

    # Bridge activation is conditional on concentrated neighborhoods.
    pair_concentration = np.sqrt(
        concentration[:, None] * concentration[None, :]
    )
    scarcity = np.sqrt(
        np.minimum(valid_degree[:, None], valid_degree[None:]) / degree_scale
    )

    # Long edges receive less bridge credit unless they have real regret.
    distance_guard = np.exp(
        -np.maximum(distances - scale, 0.0) / max(2.0 * scale, np.finfo(float).eps)
    )
    bridge_signal = (
        bridge
        * pair_concentration
        * scarcity
        * distance_guard
        * (0.35 + 0.65 * regret)
    )
    bridge_signal[~finite] = 0.0

    concentration_level = float(np.mean(concentration))
    continuation_weight = 0.15 + 0.10 * concentration_level
    regret_weight = 0.18 + 0.05 * concentration_level
    bridge_weight = 0.04 + 0.12 * concentration_level

    score = (
        0.58 * affinity
        + 0.16 * endpoint
        + regret_weight * regret
        + continuation_weight * continuation
        + bridge_weight * bridge_signal
    )
    score[~finite] = 0.0
    score = 0.5 * (score + score.T)

    # One-pass local competition control avoids excessive global smoothing.
    row_strength = score.sum(axis=1)
    strength_scale = np.median(row_strength[row_strength > 0.0]) \
        if np.any(row_strength > 0.0) else 1.0
    strength_scale = max(float(strength_scale), np.finfo(float).eps)

    node_factor = np.sqrt(
        np.clip(strength_scale / np.maximum(row_strength, strength_scale), 0.65, 1.35)
    )
    local_balance = node_factor[:, None] * node_factor[None, :]
    local_balance = np.clip(local_balance, 0.72, 1.28)

    # Protect the strongest short edges from competition suppression.
    protection = 0.35 + 0.65 * affinity
    score *= 0.82 + 0.18 * local_balance
    score += 0.035 * affinity * protection
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    result = 0.88 * score + 0.12 * affinity
    result[~finite] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum_result = float(np.max(result))
    if np.isfinite(maximum_result) and maximum_result > 0.0:
        result /= maximum_result

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
