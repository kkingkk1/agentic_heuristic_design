import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return symmetric, scale-invariant edge-priority indicators for TSP sampling.
    Larger values indicate more promising undirected edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(d)
    valid = finite & off_diagonal & (d >= 0)

    if not np.any(valid):
        return result

    positive = d[valid & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    safe_d = np.full((n, n), scale, dtype=float)
    safe_d[valid] = np.maximum(d[valid], eps)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / safe_d[valid]

    maximum = np.max(inverse)
    if maximum > 0:
        inverse /= maximum

    # Relative attractiveness of an edge at each endpoint.
    row_sum = np.sum(inverse, axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    endpoint_share = inverse / row_sum

    # Edges dominant at both endpoints receive a larger score.
    selectivity = np.sqrt(endpoint_share * endpoint_share.T)

    # Symmetric short-edge strength.
    reciprocal_strength = 0.5 * (inverse + inverse.T)

    score = reciprocal_strength * (0.5 + 0.5 * selectivity)

    # Preserve only originally valid off-diagonal edges.
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
