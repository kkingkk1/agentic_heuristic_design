import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    symmetric_d = np.zeros((n, n), dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])
    symmetric_d = np.maximum(symmetric_d, 0.0)

    positive_values = symmetric_d[valid & (symmetric_d > 0.0)]
    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0
    scale = max(scale, eps)

    normalized_cost = np.divide(
        symmetric_d,
        scale,
        out=np.zeros_like(symmetric_d),
        where=valid,
    )
    normalized_cost = np.minimum(normalized_cost, 60.0)

    # Global Gibbs affinity, with no hard candidate truncation.
    temperature = 0.85
    affinity = np.exp(-normalized_cost / temperature)
    affinity = np.where(valid, affinity, 0.0)
    np.fill_diagonal(affinity, 0.0)

    # Symmetric matrix scaling approximates a degree-balanced soft 2-factor.
    balanced = affinity.copy()
    for _ in range(24):
        row_sum = np.sum(balanced, axis=1)
        row_sum = np.maximum(row_sum, eps)
        scale_vector = 1.0 / np.sqrt(row_sum)
        balanced = (
            scale_vector[:, None]
            * balanced
            * scale_vector[None, :]
        )
        balanced = 0.5 * (balanced + balanced.T)
        balanced = np.where(valid, balanced, 0.0)
        np.fill_diagonal(balanced, 0.0)

    balanced_max = float(np.max(balanced))
    if balanced_max > 0.0 and np.isfinite(balanced_max):
        balanced_unit = balanced / balanced_max
    else:
        balanced_unit = np.zeros_like(balanced)

    # Two-hop support measures whether endpoints share short alternatives.
    two_hop = balanced @ balanced
    two_hop = 0.5 * (two_hop + two_hop.T)
    two_hop = np.where(valid, two_hop, 0.0)
    np.fill_diagonal(two_hop, 0.0)

    support_scale = np.sqrt(
        np.maximum(
            np.sum(balanced * balanced, axis=1)[:, None]
            * np.sum(balanced * balanced, axis=1)[None, :],
            eps,
        )
    )
    normalized_support = np.divide(
        two_hop,
        support_scale,
        out=np.zeros_like(two_hop),
        where=support_scale > 0.0,
    )
    normalized_support = np.clip(normalized_support, 0.0, 1.0)

    # Preserve direct affinity while adding a structural closure signal.
    score = balanced_unit * (
        0.72 + 0.28 * normalized_support
    )
    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
