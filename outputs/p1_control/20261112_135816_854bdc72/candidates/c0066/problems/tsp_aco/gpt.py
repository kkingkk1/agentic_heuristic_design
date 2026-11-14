import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return symmetric edge-prior indicators for stochastic TSP sampling.

    Larger values indicate edges that are more promising.
    The output has the same shape as distance_matrix.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(d)
    valid_values = d[finite & off_diagonal]

    if valid_values.size == 0:
        return np.zeros_like(d, dtype=float)

    positive_values = valid_values[valid_values > 0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    safe_d = np.where(np.isfinite(d), np.maximum(d, 0.0), np.inf)
    safe_d[~off_diagonal] = np.inf

    row_values = np.where(np.isfinite(safe_d), safe_d, np.nan)
    row_scale = np.nanmedian(
        np.where(row_values > 0, row_values, np.nan),
        axis=1,
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        global_scale,
    )

    normalized = safe_d / np.sqrt(
        row_scale[:, None] * row_scale[None, :]
    )
    normalized = np.where(np.isfinite(normalized), normalized, np.inf)

    # Short-distance attractiveness.
    short_edge = np.exp(-np.minimum(normalized, 50.0))

    # Mutual-nearest-neighbor compatibility.
    nearest = np.min(safe_d, axis=1)
    nearest = np.maximum(nearest, eps)
    relative_distance = (
        safe_d / nearest[:, None]
        + safe_d / nearest[None, :]
    )
    endpoint_affinity = np.exp(-np.minimum(relative_distance, 50.0))

    # Endpoint scarcity based on within-row distance ranks.
    row_rank = np.argsort(
        np.argsort(
            np.where(np.isfinite(safe_d), safe_d, np.inf),
            axis=1,
        ),
        axis=1,
    )
    rank_score = 1.0 - row_rank / max(n - 1, 1)
    scarcity = np.sqrt(
        np.maximum(rank_score, 0.0)
        * np.maximum(rank_score.T, 0.0)
    )

    # Soft bottleneck penalty: discount edges that consume unusually
    # attractive alternatives at either endpoint.
    local_attractiveness = np.exp(-np.minimum(normalized, 18.0))
    endpoint_load = np.sum(local_attractiveness, axis=1)
    endpoint_load = np.maximum(endpoint_load, eps)
    edge_load = 0.5 * (
        endpoint_load[:, None] + endpoint_load[None, :]
    )
    bottleneck_penalty = np.exp(
        -0.16 * np.minimum(edge_load, 30.0)
    )

    # Mild diversity correction for dense endpoints.
    competition = np.sum(
        np.exp(-np.minimum(normalized, 16.0)),
        axis=1,
    )
    competition = np.maximum(competition, eps)
    diversity = 1.0 / np.sqrt(
        competition[:, None] * competition[None, :]
    )

    score = (
        np.maximum(short_edge, 1e-12) ** 0.48
        * np.maximum(endpoint_affinity, 1e-12) ** 0.24
        * np.maximum(scarcity, 1e-12) ** 0.18
        * np.maximum(bottleneck_penalty, 1e-12) ** 0.06
        * np.maximum(diversity, 1e-12) ** 0.04
    )

    score[~off_diagonal] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
