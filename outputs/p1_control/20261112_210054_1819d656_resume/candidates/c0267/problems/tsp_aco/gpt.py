import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances)
    positive = distances[finite & off_diagonal & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    reference = float(np.median(positive))
    if not np.isfinite(reference) or reference <= 0.0:
        reference = float(np.mean(positive))
    if not np.isfinite(reference) or reference <= 0.0:
        reference = 1.0

    upper_quantile = float(np.percentile(positive, 75.0))
    if not np.isfinite(upper_quantile) or upper_quantile <= 0.0:
        upper_quantile = reference

    replacement = max(reference, upper_quantile)

    d = np.where(finite, distances, replacement)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, replacement)

    positive_offdiag = d[off_diagonal & (d > 0.0)]
    if positive_offdiag.size:
        scale = float(np.percentile(positive_offdiag, 60.0))
    else:
        scale = reference

    if not np.isfinite(scale) or scale <= 0.0:
        scale = reference

    eps = np.finfo(float).eps

    # Bounded reciprocal affinity avoids the exponential/rank constructions
    # used by the parent heuristics.
    normalized = d / (scale + eps)
    affinity = 1.0 / (1.0 + normalized)
    affinity[~off_diagonal] = 0.0

    # Remove a uniform background so very long repaired edges contribute little.
    background = float(np.percentile(affinity[off_diagonal], 15.0))
    affinity = np.maximum(affinity - background, 0.0)
    affinity[~off_diagonal] = 0.0

    if not np.any(affinity > 0.0):
        return np.zeros((n, n), dtype=float)

    # Entropic global balancing: repeatedly enforce comparable row and column
    # masses, producing a globally competitive rather than locally ranked score.
    balanced = affinity.copy()
    for _ in range(24):
        row_mass = balanced.sum(axis=1)
        row_mass = np.maximum(row_mass, eps)
        balanced /= row_mass[:, None]

        col_mass = balanced.sum(axis=0)
        col_mass = np.maximum(col_mass, eps)
        balanced /= col_mass[None, :]

        balanced = 0.5 * (balanced + balanced.T)
        balanced[~off_diagonal] = 0.0

    balanced = np.maximum(balanced, 0.0)

    # Residual endpoint capacity rewards edges whose two endpoints still have
    # concentrated support after global balancing.
    row_concentration = np.sum(balanced * balanced, axis=1)
    row_concentration = np.maximum(row_concentration, eps)

    endpoint_capacity = np.sqrt(
        row_concentration[:, None] * row_concentration[None, :]
    )
    endpoint_capacity /= np.maximum(
        np.max(endpoint_capacity),
        eps,
    )

    # A mild global cost factor distinguishes edges with similar balanced mass.
    global_cost_factor = 1.0 / (
        1.0 + np.sqrt(np.maximum(d, 0.0) / (reference + eps))
    )

    score = balanced * (0.72 + 0.28 * endpoint_capacity)
    score *= 0.78 + 0.22 * global_cost_factor

    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
