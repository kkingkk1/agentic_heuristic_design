import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric TSP edge prior based on soft two-hop substitution."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    symmetric_valid = valid & valid.T
    positive = distances[valid & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.percentile(positive, 75.0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    upper_fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(upper_fallback) or upper_fallback <= 0.0:
        upper_fallback = float(np.max(positive))
    if not np.isfinite(upper_fallback) or upper_fallback <= 0.0:
        upper_fallback = scale

    repaired = np.where(valid, distances, upper_fallback)
    repaired = np.maximum(repaired, 0.0)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 0.0)

    edge_mask = symmetric_valid.astype(float)
    eps = np.finfo(float).eps

    normalized = repaired / max(scale, eps)

    direct_affinity = 1.0 / (1.0 + normalized)
    direct_affinity *= edge_mask
    np.fill_diagonal(direct_affinity, 0.0)

    row_mass = np.sum(direct_affinity, axis=1, keepdims=True)
    transition = direct_affinity / np.maximum(row_mass, eps)

    two_hop_support = transition @ transition.T
    two_hop_support = np.maximum(two_hop_support, 0.0)

    direct_mass = direct_affinity / np.maximum(
        np.sum(direct_affinity, axis=1, keepdims=True),
        eps,
    )
    endpoint_balance = np.sqrt(
        np.maximum(direct_mass * direct_mass.T, 0.0)
    )

    substitution_ratio = direct_affinity / (
        direct_affinity + two_hop_support + 0.25
    )
    substitution_ratio = np.clip(substitution_ratio, 0.0, 1.0)

    bridge_bonus = 0.55 + 0.45 * substitution_ratio
    balance_bonus = 0.70 + 0.30 * endpoint_balance

    score = (
        direct_affinity
        * bridge_bonus
        * balance_bonus
    )

    score = 0.5 * (score + score.T)
    score *= edge_mask
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
