import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-structure-aware edge priorities for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0.0)
    valid = finite_positive & finite_positive.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distance = np.zeros_like(d, dtype=float)
    symmetric_distance[valid] = 0.5 * (
        d[valid] + d.T[valid]
    )

    eps = np.finfo(float).eps
    positive_values = symmetric_distance[valid]
    global_scale = max(float(np.median(positive_values)), eps)

    normalized_distance = np.zeros_like(symmetric_distance, dtype=float)
    normalized_distance[valid] = (
        symmetric_distance[valid] / global_scale
    )
    normalized_distance = np.maximum(normalized_distance, eps)

    # Bounded direct affinity avoids exponential collapse for long but useful edges.
    direct_affinity = np.zeros_like(symmetric_distance, dtype=float)
    direct_affinity[valid] = 1.0 / (
        1.0 + normalized_distance[valid]
    )

    # Soft endpoint choice probabilities over all feasible incident edges.
    row_strength = np.sum(direct_affinity, axis=1)
    safe_row_strength = np.maximum(row_strength, eps)

    choice_probability = direct_affinity / safe_row_strength[:, None]
    reciprocal_choice = np.sqrt(
        choice_probability * choice_probability.T
    )

    # Estimate whether an edge has many attractive two-hop substitutes.
    two_hop_support = direct_affinity @ direct_affinity
    two_hop_support[~valid] = 0.0

    support_values = two_hop_support[valid]
    support_scale = (
        max(float(np.median(support_values)), eps)
        if support_values.size
        else 1.0
    )
    normalized_support = two_hop_support / support_scale

    # Low redundancy mildly favors bridge-like edges needed for connectivity.
    bridge_factor = np.zeros_like(symmetric_distance, dtype=float)
    bridge_factor[valid] = 1.0 / (
        1.0 + normalized_support[valid]
    )

    # Avoid strongly favoring edges between nodes with very different connectivity.
    strength_ratio = (
        row_strength[:, None] + eps
    ) / (
        row_strength[None, :] + eps
    )
    strength_imbalance = np.abs(np.log(strength_ratio))
    balance_factor = np.zeros_like(symmetric_distance, dtype=float)
    balance_factor[valid] = 1.0 / (
        1.0 + 0.35 * np.minimum(strength_imbalance[valid], 50.0)
    )

    score = (
        np.power(np.maximum(direct_affinity, eps), 0.80)
        * np.power(np.maximum(reciprocal_choice, eps), 1.25)
        * np.power(np.maximum(bridge_factor, eps), 0.30)
        * np.power(np.maximum(balance_factor, eps), 0.25)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
