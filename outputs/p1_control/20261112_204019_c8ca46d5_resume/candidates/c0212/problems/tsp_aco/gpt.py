import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return random-walk and two-hop structural priorities for TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    if positive.size:
        fallback = float(np.max(positive))
    else:
        fallback = 1.0

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)

    np.fill_diagonal(d, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    positive_off_diagonal = d[off_diagonal & (d > 0.0)]

    if positive_off_diagonal.size:
        scale = float(np.median(positive_off_diagonal))
    else:
        scale = fallback

    eps = np.finfo(float).eps
    scale = max(scale, eps)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    inverse_affinity = 1.0 / (
        1.0 + np.maximum(work, 0.0) / scale
    )
    inverse_affinity[~np.isfinite(inverse_affinity)] = 0.0
    np.fill_diagonal(inverse_affinity, 0.0)

    row_totals = inverse_affinity.sum(axis=1, keepdims=True)
    row_totals = np.maximum(row_totals, eps)
    transition = inverse_affinity / row_totals

    reciprocal_support = np.sqrt(
        transition * transition.T
    )

    two_hop_support = transition @ transition.T
    two_hop_support = 0.5 * (
        two_hop_support + two_hop_support.T
    )
    np.fill_diagonal(two_hop_support, 0.0)

    support_values = two_hop_support[off_diagonal]
    if support_values.size:
        support_scale = float(np.percentile(support_values, 90.0))
    else:
        support_scale = 1.0

    support_scale = max(support_scale, eps)
    structural_support = np.minimum(
        two_hop_support / support_scale,
        1.0,
    )

    probabilities = np.clip(transition, eps, 1.0)
    entropy = -np.sum(
        probabilities * np.log(probabilities),
        axis=1,
    )
    entropy_limit = np.log(max(n - 1, 1))
    concentration = 1.0 - entropy / max(entropy_limit, eps)
    concentration = np.clip(concentration, 0.0, 1.0)

    concentration_pair = np.sqrt(
        concentration[:, None] * concentration[None, :]
    )

    direct_support = np.sqrt(
        inverse_affinity * inverse_affinity.T
    )
    direct_max = float(np.max(direct_support))

    if np.isfinite(direct_max) and direct_max > 0.0:
        direct_support /= direct_max

    score = (
        0.42 * direct_support
        + 0.33 * reciprocal_support
        + 0.25 * structural_support
    )

    score *= 0.82 + 0.18 * concentration_pair

    score = 0.5 * (score + score.T)
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
