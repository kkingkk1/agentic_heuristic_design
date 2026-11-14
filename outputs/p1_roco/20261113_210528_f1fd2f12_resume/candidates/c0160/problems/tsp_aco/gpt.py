import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structure-aware edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    positive = distances[valid & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)
    eps = np.finfo(float).eps * max(scale, 1.0)

    safe = np.where(valid, distances, np.inf)

    # Short-distance attraction with row-wise endpoint competition.
    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + eps)

    row_peak = np.maximum(np.max(inverse, axis=1, keepdims=True), eps)
    endpoint_preference = inverse / row_peak

    # Mutual rank agreement between the two endpoints.
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.full((n, n), float(n))
    rank_values = np.arange(n, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = rank_values

    rank_score = np.zeros((n, n), dtype=float)
    rank_score[valid] = 1.0 / (1.0 + ranks[valid])
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    # Smooth local support and node scarcity.
    proximity = np.zeros((n, n), dtype=float)
    proximity[valid] = np.exp(-safe[valid] / (0.85 * scale))

    row_mass = proximity.sum(axis=1)
    local_preference = proximity / np.maximum(row_mass[:, None], eps)
    mutual_support = np.sqrt(
        np.maximum(local_preference * local_preference.T, 0.0)
    )

    scarcity = 1.0 / np.maximum(row_mass, eps)
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity_support = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    scarcity_support = np.clip(scarcity_support, 0.0, 2.5)

    # Two-hop detour advantage: high values indicate irreplaceable edges.
    best_two_hop = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        candidate = safe[:, k, None] + safe[None, k, :]
        best_two_hop = np.minimum(best_two_hop, candidate)

    np.fill_diagonal(best_two_hop, np.inf)

    ratio = np.zeros((n, n), dtype=float)
    bypass_exists = valid & np.isfinite(best_two_hop)
    ratio[bypass_exists] = (
        best_two_hop[bypass_exists]
        / np.maximum(distances[bypass_exists], eps)
    )

    irreplaceability = np.ones((n, n), dtype=float)
    irreplaceability[bypass_exists] = np.tanh(
        np.maximum(ratio[bypass_exists] - 1.0, 0.0)
    )
    irreplaceability = 0.20 + 0.80 * irreplaceability

    # Multiplicative fusion preserves agreement across complementary signals.
    affinity = inverse / np.maximum(np.max(inverse), eps)
    affinity = 0.10 + 0.90 * affinity

    score = (
        np.power(np.maximum(affinity, eps), 0.42)
        * np.power(0.20 + 0.80 * endpoint_preference, 0.18)
        * np.power(0.20 + 0.80 * mutual_rank, 0.16)
        * np.power(0.65 + 0.35 * mutual_support, 0.10)
        * np.power(0.70 + 0.30 * scarcity_support, 0.06)
        * np.power(irreplaceability, 0.08)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
