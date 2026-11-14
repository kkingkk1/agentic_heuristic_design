import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    if np.any(np.isnan(d)) or np.any(d < 0):
        raise ValueError("distance_matrix must contain nonnegative, non-NaN values")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    off_diag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(d)

    if not np.any(finite & off_diag):
        return np.zeros_like(d, dtype=float)

    finite_values = d[finite & off_diag]
    fallback = float(np.max(finite_values)) if finite_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    d = np.where(finite, d, fallback * 10.0)
    d = np.minimum(d, d.T)
    np.fill_diagonal(d, np.inf)

    sorted_d = np.sort(d, axis=1)
    usable = min(4, n - 1)
    local_scale = np.mean(sorted_d[:, :usable], axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    pair_scale = local_scale[:, None] + local_scale[None, :]
    normalized_distance = d / pair_scale
    sqrt_distance = np.sqrt(np.maximum(normalized_distance, 0.0))

    row_rank = np.argsort(np.argsort(d, axis=1), axis=1).astype(float)
    rank_score = 1.0 / np.power(1.0 + row_rank, 0.80)
    mutuality = np.sqrt(rank_score * rank_score.T)

    distance_score = np.exp(-1.30 * sqrt_distance)

    isolation_ratio = (
        local_scale[:, None] / pair_scale
        + local_scale[None, :] / pair_scale
    )
    isolation_bonus = 1.0 + 0.75 * isolation_ratio

    indicators = (
        distance_score
        * (0.35 + 0.65 * mutuality)
        * isolation_bonus
    )

    indicators = np.nan_to_num(
        indicators,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    indicators = 0.5 * (indicators + indicators.T)
    np.fill_diagonal(indicators, 0.0)

    maximum = np.max(indicators)
    if maximum > 0:
        indicators /= maximum

    return indicators.astype(float)
