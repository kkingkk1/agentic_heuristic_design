import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    off_diagonal = ~np.eye(n, dtype=bool)

    if not np.any(finite & off_diagonal):
        return np.zeros((n, n), dtype=float)

    d = np.where(finite, np.maximum(distances, 0.0), np.inf)
    np.fill_diagonal(d, np.inf)

    valid_values = d[np.isfinite(d)]
    positive_values = valid_values[valid_values > 0.0]
    base_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.median(valid_values)), 1.0)
    )
    base_scale = max(base_scale, np.finfo(float).eps)

    row_medians = np.full(n, base_scale, dtype=float)
    for i in range(n):
        row_values = d[i, np.isfinite(d[i])]
        if row_values.size:
            row_medians[i] = max(float(np.median(row_values)), base_scale * 1e-6)

    endpoint_scale = np.sqrt(
        row_medians[:, None] * row_medians[None, :]
    )
    endpoint_scale = np.maximum(endpoint_scale, np.finfo(float).eps)

    # Soft preference for edges that are cheap relative to both endpoints.
    relative_cost = d / endpoint_scale
    local_preference = 1.0 / (1.0 + np.square(relative_cost))

    # Estimate the best detour replacing edge (i, j) through an intermediate node.
    two_hop = d[:, :, None] + d[None, :, :]
    np.fill_diagonal(two_hop[:, :, :], np.inf)
    best_detour = np.min(two_hop, axis=2)

    detour_margin = best_detour - d
    margin_scale = base_scale + endpoint_scale

    # Saturating transformation rewards difficult-to-bypass edges.
    bridge_value = 0.5 + 0.5 * np.tanh(detour_margin / margin_scale)

    score = local_preference * bridge_value
    score = (score + score.T) / 2.0
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
