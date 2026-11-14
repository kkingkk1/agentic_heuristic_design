import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric topology-aware prior for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_positive = np.isfinite(raw) & (raw > 0.0)
    valid = finite_positive & finite_positive.T

    if not np.any(valid):
        return result

    distances = np.zeros_like(raw, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])

    positive = distances[valid]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive)), eps)

    row_values = np.where(valid, distances, np.nan)
    row_center = np.nanmedian(row_values, axis=1)
    row_center = np.where(
        np.isfinite(row_center) & (row_center > 0.0),
        row_center,
        global_scale,
    )

    deviations = np.abs(row_values - row_center[:, None])
    row_spread = np.nanmedian(deviations, axis=1)
    row_spread = np.where(
        np.isfinite(row_spread) & (row_spread > eps),
        row_spread,
        0.25 * global_scale,
    )
    row_spread = np.maximum(row_spread, eps)

    pair_center = 0.5 * (
        row_center[:, None] + row_center[None, :]
    )
    pair_spread = 0.5 * (
        row_spread[:, None] + row_spread[None, :]
    )

    standardized = (distances - pair_center) / (
        pair_spread + 0.1 * global_scale + eps
    )
    affinity = np.exp(-0.5 * np.square(np.clip(standardized, -12.0, 12.0)))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_mass = np.sum(affinity, axis=1, keepdims=True)
    transition = affinity / np.maximum(row_mass, eps)

    two_hop_support = transition @ transition
    two_hop_support = np.maximum(two_hop_support, 0.0)

    degree_balance = 1.0 / np.sqrt(
        np.maximum(row_mass, eps) * np.maximum(row_mass.T, eps)
    )
    degree_balance /= max(float(np.max(degree_balance[valid])), eps)
    degree_factor = 0.75 + 0.25 * degree_balance

    direct_support = np.sqrt(
        transition * transition.T
    )

    score = (
        direct_support
        * (0.55 + 0.45 * np.sqrt(two_hop_support))
        * degree_factor
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
