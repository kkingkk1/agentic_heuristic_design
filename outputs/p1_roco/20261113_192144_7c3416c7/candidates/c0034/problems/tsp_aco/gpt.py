import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priors based on two-hop replacement savings."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    costs = np.where(valid, np.maximum(distances, 0.0), np.inf)
    costs = 0.5 * (costs + costs.T)
    valid = np.isfinite(costs)
    np.fill_diagonal(costs, np.inf)

    edge_values = costs[valid]
    scale = float(np.median(edge_values)) if edge_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    two_hop = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        left = costs[:, k][:, None]
        right = costs[k, :][None, :]
        candidate = left + right

        candidate[k, :] = np.inf
        candidate[:, k] = np.inf

        two_hop = np.minimum(two_hop, candidate)

    direct = costs
    finite_alternative = np.isfinite(two_hop)

    savings = np.zeros((n, n), dtype=float)
    savings[finite_alternative] = np.maximum(
        two_hop[finite_alternative] - direct[finite_alternative],
        0.0,
    )

    fallback = ~finite_alternative & np.isfinite(direct)
    savings[fallback] = scale

    normalized_savings = savings / (savings + scale)

    direct_scale = np.maximum(direct, np.finfo(float).eps)
    replacement_ratio = np.zeros((n, n), dtype=float)
    replacement_ratio[finite_alternative] = np.clip(
        savings[finite_alternative] / direct_scale[finite_alternative],
        0.0,
        1e6,
    )
    replacement_support = replacement_ratio / (1.0 + replacement_ratio)

    endpoint_difficulty = np.zeros(n, dtype=float)
    for i in range(n):
        alternatives = two_hop[i]
        finite_alternatives = alternatives[np.isfinite(alternatives)]
        if finite_alternatives.size:
            endpoint_difficulty[i] = np.median(finite_alternatives)

    difficulty_scale = np.median(
        endpoint_difficulty[endpoint_difficulty > 0]
    ) if np.any(endpoint_difficulty > 0) else scale

    endpoint_factor = np.sqrt(
        endpoint_difficulty[:, None] * endpoint_difficulty[None, :]
    )
    endpoint_factor = endpoint_factor / (
        endpoint_factor + max(float(difficulty_scale), np.finfo(float).eps)
    )

    score = (
        0.55 * normalized_savings
        + 0.30 * replacement_support
        + 0.15 * endpoint_factor
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(direct)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
