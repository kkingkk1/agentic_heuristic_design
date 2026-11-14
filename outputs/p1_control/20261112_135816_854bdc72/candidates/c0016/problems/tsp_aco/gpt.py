import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-prior indicators for stochastic TSP solution sampling.

    Larger values indicate edges that are more promising to include.
    The output has the same shape as distance_matrix.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    off_diagonal = ~np.eye(n, dtype=bool)

    valid_values = d[finite & off_diagonal]
    if valid_values.size == 0:
        return np.zeros_like(d, dtype=float)

    positive_values = valid_values[valid_values > 0]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe_d = np.where(np.isfinite(d), np.maximum(d, 0.0), np.inf)
    safe_d[np.eye(n, dtype=bool)] = np.inf

    # Robust distance scale for each endpoint.
    row_values = np.where(np.isfinite(safe_d), safe_d, np.nan)
    row_scale = np.nanmedian(
        np.where(row_values > 0, row_values, np.nan),
        axis=1
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        scale
    )

    normalized = safe_d / np.sqrt(row_scale[:, None] * row_scale[None, :])
    normalized = np.where(np.isfinite(normalized), normalized, np.inf)

    # Local attractiveness: short edges receive exponentially larger scores.
    short_edge = np.exp(-np.minimum(normalized, 60.0))

    # Mutual-nearest-neighbor compatibility.
    nearest = np.min(
        np.where(np.isfinite(safe_d), safe_d, np.inf),
        axis=1
    )
    nearest = np.maximum(nearest, np.finfo(float).eps)
    endpoint_affinity = np.exp(
        -np.minimum(
            safe_d / nearest[:, None] + safe_d / nearest[None, :],
            60.0
        )
    )

    # Scarcity: preserve edges that are relatively important to both endpoints.
    row_rank = np.argsort(np.argsort(np.where(np.isfinite(safe_d), safe_d, np.inf), axis=1), axis=1)
    rank_score = 1.0 - row_rank / max(n - 1, 1)
    scarcity = np.sqrt(np.maximum(rank_score, 0.0) * np.maximum(rank_score.T, 0.0))

    # Penalize edges whose endpoints already have many similarly attractive choices.
    competition = np.sum(np.exp(-np.minimum(normalized, 20.0)), axis=1)
    competition = np.maximum(competition, np.finfo(float).eps)
    diversity = 1.0 / np.sqrt(competition[:, None] * competition[None, :])

    score = (
        np.maximum(short_edge, 1e-12) ** 0.45
        * np.maximum(endpoint_affinity, 1e-12) ** 0.25
        * np.maximum(scarcity, 1e-12) ** 0.20
        * np.maximum(diversity, 1e-12) ** 0.10
    )

    score[~off_diagonal] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
