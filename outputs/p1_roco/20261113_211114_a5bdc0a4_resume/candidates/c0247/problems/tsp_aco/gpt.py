import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, diversified edge-selection priors for stochastic TSP sampling."""
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
    valid = finite & finite.T

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

    q95 = float(np.percentile(observed, 95.0)) if observed.size else scale
    cap = max(q95 + 4.0 * scale, scale)
    distances = np.minimum(distances, cap)
    distances[~valid] = cap
    np.fill_diagonal(distances, cap)

    temperature = max(0.72 * scale, 0.20 * cap, np.finfo(float).eps)
    affinity = np.exp(-distances / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    maximum = float(np.max(affinity))
    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0
    affinity /= maximum

    local_rank = np.zeros((n, n), dtype=float)
    node_entropy = np.zeros(n, dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[
            np.argsort(distances[i, candidates], kind="stable")
        ]
        denominator = max(float(order.size - 1), 1.0)
        local_rank[i, order] = 1.0 - np.arange(order.size) / denominator

        row_affinity = affinity[i, candidates]
        mass = float(row_affinity.sum())
        if mass > 0.0 and candidates.size > 1:
            probabilities = row_affinity / mass
            node_entropy[i] = float(
                -np.sum(probabilities * np.log(probabilities + 1e-12))
                / np.log(float(candidates.size))
            )

    local_rank = np.sqrt(local_rank * local_rank.T)
    local_rank[~valid] = 0.0

    # Limited two-hop support preserves useful continuation information.
    transition = affinity + 0.10 * local_rank
    transition[~valid] = 0.0
    row_sum = transition.sum(axis=1, keepdims=True)
    transition /= np.maximum(row_sum, np.finfo(float).eps)

    two_hop = transition @ transition
    two_hop = 0.5 * (two_hop + two_hop.T)
    two_hop[~valid] = 0.0
    two_hop_max = float(np.max(two_hop))
    if two_hop_max > 0.0 and np.isfinite(two_hop_max):
        two_hop /= two_hop_max

    # Mild diversity bonus only for concentrated local neighborhoods.
    scarcity = np.clip(1.0 - node_entropy, 0.0, 1.0)
    diversity = np.sqrt(scarcity[:, None] * scarcity[None, :])
    diversity[~valid] = 0.0

    score = (
        0.78 * affinity
        + 0.12 * local_rank
        + 0.07 * two_hop
        + 0.03 * diversity
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    # Finite symmetric balancing reduces node concentration without dominating distance.
    balanced = score.copy()
    for _ in range(18):
        row_sum = balanced.sum(axis=1, keepdims=True)
        normalized = balanced / np.maximum(row_sum, np.finfo(float).eps)
        balanced = 0.78 * balanced + 0.22 * normalized
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    result = 0.86 * score + 0.14 * balanced
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
