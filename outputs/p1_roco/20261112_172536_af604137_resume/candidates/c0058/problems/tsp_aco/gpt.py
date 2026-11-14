import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = distances.copy()
    d[~np.isfinite(d)] = np.inf
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0]
    fallback = float(np.max(positive_values)) if positive_values.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)
    d[~np.isfinite(d)] = fallback * 10.0

    positive = d[d > 0]
    global_scale = float(np.median(positive)) if positive.size else fallback
    global_scale = max(global_scale, np.finfo(float).eps)

    # Estimate a local distance scale from the closest non-self neighbors.
    k = min(max(2, int(np.sqrt(n))), n - 1)
    sorted_distances = np.sort(d, axis=1)
    local_scale = np.mean(sorted_distances[:, :k], axis=1)
    local_scale = np.maximum(local_scale, 0.05 * global_scale)

    # Smooth inverse-distance preference with node-specific scaling.
    inverse = 1.0 / (d + 0.08 * local_scale[:, None])
    inverse /= np.mean(inverse, axis=1, keepdims=True) + 1e-12

    # Local soft-neighborhood preference.
    local_soft = np.exp(-d / (local_scale[:, None] + 1e-12))
    local_soft /= np.mean(local_soft, axis=1, keepdims=True) + 1e-12

    # Rank-based preference, with a slower decay for larger instances.
    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_scale = max(1.5, 0.18 * n)
    rank_score = np.exp(-ranks / rank_scale)

    # Mutual nearest-neighborhood preference.
    nearest = np.argmin(d, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    mutual[np.arange(n), nearest] = 1.0
    mutual[nearest, np.arange(n)] = 1.0

    # Bidirectional relative closeness.
    relative = np.sqrt(
        np.exp(-d / (local_scale[:, None] + 1e-12))
        * np.exp(-d / (local_scale[None, :] + 1e-12))
    )

    # Mild suppression of edges that are unusually long for either endpoint.
    row_median = np.median(d, axis=1)
    excess = np.maximum(
        0.0,
        d - np.maximum(row_median[:, None], local_scale[:, None]),
    )
    long_edge_gate = 1.0 / (1.0 + excess / (global_scale + 1e-12))

    # Combine directional evidence before enforcing symmetry.
    directional = (
        0.30 * inverse
        + 0.24 * local_soft
        + 0.20 * rank_score
        + 0.16 * relative
        + 0.10 * long_edge_gate
    )

    # Symmetric geometric averaging rewards edges attractive from both ends.
    score = np.sqrt(
        np.maximum(directional, 0.0)
        * np.maximum(directional.T, 0.0)
    )

    # Preserve explicit mutual-nearest-neighbor structure without making it dominant.
    score += 0.08 * mutual

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
