import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_positive = np.isfinite(d) & (d > 0)
    valid = finite_positive & finite_positive.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    replacement = max(float(np.max(positive_values)) * 10.0, global_scale)

    safe_d = np.where(valid, 0.5 * (d + d.T), replacement)
    np.fill_diagonal(safe_d, replacement)

    # Local scale from the third-nearest valid neighbor when available.
    neighbor_count = min(3, n - 1)
    kth_index = min(neighbor_count - 1, n - 2)
    local_scale = np.partition(safe_d, kth_index, axis=1)[:, kth_index]
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        np.outer(local_scale, local_scale)
    )

    # Shortness relative to both endpoint neighborhoods.
    normalized_distance = safe_d / endpoint_scale
    local_closeness = np.exp(-np.clip(0.9 * normalized_distance, 0.0, 745.0))

    # Robust global distance preference.
    global_closeness = global_scale / (global_scale + safe_d)

    # Mutual endpoint rank preference.
    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_scale = max(float(n - 1), 1.0)
    rank_score = np.exp(
        -2.0 * (ranks / rank_scale + ranks.T / rank_scale)
    )

    # Compatibility between the local scales of the two endpoints.
    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )
    scale_compatibility = np.exp(-0.8 * np.clip(scale_difference, 0.0, 1.0))

    # Mildly suppress edges incident to excessively crowded neighborhoods.
    neighborhood_density = np.sum(
        np.exp(-np.clip(safe_d / local_scale[:, None], 0.0, 745.0))
        * valid,
        axis=1,
    )
    crowding_penalty = 1.0 / np.sqrt(
        (1.0 + neighborhood_density[:, None])
        * (1.0 + neighborhood_density[None, :])
    )

    score = (
        0.35 * global_closeness
        + 0.35 * local_closeness
        + 0.20 * rank_score
        + 0.10 * scale_compatibility
    )
    score *= crowding_penalty

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
