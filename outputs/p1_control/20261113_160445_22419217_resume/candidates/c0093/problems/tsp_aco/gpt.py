import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a graph-diffusion prior for stochastic TSP edge sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    safe = np.where(valid, distances, np.inf)

    # Estimate a robust local bandwidth for each node.
    local_values = np.where(valid, distances, np.nan)
    local_median = np.nanmedian(local_values, axis=1)

    positive = distances[valid & (distances > 0.0)]
    global_scale = (
        float(np.median(positive)) if positive.size else 1.0
    )
    global_scale = max(global_scale, eps)

    local_median = np.where(
        np.isfinite(local_median) & (local_median > 0.0),
        local_median,
        global_scale,
    )

    local_spread = np.nanmedian(
        np.abs(local_values - local_median[:, None]),
        axis=1,
    )
    local_spread = np.where(
        np.isfinite(local_spread) & (local_spread > 0.0),
        local_spread,
        0.5 * global_scale,
    )
    local_spread = np.maximum(local_spread, eps)

    bandwidth = np.sqrt(
        local_spread[:, None] * local_spread[None, :]
    )
    bandwidth = np.maximum(bandwidth, eps)

    # Adaptive Gaussian affinity graph.
    normalized = np.divide(
        distances,
        bandwidth,
        out=np.zeros_like(distances),
        where=valid,
    )
    affinity = np.exp(-0.5 * np.minimum(normalized * normalized, 80.0))
    affinity = np.where(valid, affinity, 0.0)
    affinity = 0.5 * (affinity + affinity.T)

    # Normalize edge influence by endpoint weighted degree.
    degree = affinity.sum(axis=1)
    degree_scale = np.sqrt(
        np.maximum(degree[:, None] * degree[None, :], eps)
    )
    normalized_graph = affinity / degree_scale
    normalized_graph[~valid] = 0.0
    np.fill_diagonal(normalized_graph, 0.0)

    # Two-hop neighborhood compatibility and closure support.
    closure = normalized_graph @ normalized_graph
    closure = 0.5 * (closure + closure.T)
    closure[~valid] = 0.0
    np.fill_diagonal(closure, 0.0)

    # Penalize edges attached to overly diffuse neighborhoods.
    concentration = np.sum(
        np.square(normalized_graph),
        axis=1,
    )
    concentration = np.clip(concentration, 0.0, 1.0)
    endpoint_focus = np.sqrt(
        concentration[:, None] * concentration[None, :]
    )

    score = (
        0.22 * affinity
        + 0.78 * closure * (0.5 + 0.5 * endpoint_focus)
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    # Preserve a positive fallback for valid edges in graph regions
    # without two-hop support.
    for i in range(n):
        if np.any(valid[i]) and not np.any(score[i] > 0.0):
            candidates = np.flatnonzero(valid[i])
            best = candidates[np.argmin(distances[i, candidates])]
            score[i, best] = affinity[i, best]
            score[best, i] = score[i, best]

    score[~np.isfinite(score)] = 0.0
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
