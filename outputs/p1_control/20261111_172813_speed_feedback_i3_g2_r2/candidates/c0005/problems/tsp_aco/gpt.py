import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return symmetric edge-priority indicators for stochastic TSP sampling.
    Larger values indicate more promising undirected edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    nonnegative = d >= 0.0
    usable_directed = finite & nonnegative
    np.fill_diagonal(usable_directed, False)

    # Build a symmetric distance representation from available directions.
    symmetric_valid = usable_directed | usable_directed.T
    np.fill_diagonal(symmetric_valid, False)

    if not np.any(symmetric_valid):
        return np.zeros_like(d, dtype=float)

    reverse = d.T
    both = usable_directed & usable_directed.T

    symmetric_distance = np.zeros((n, n), dtype=float)
    symmetric_distance[both] = 0.5 * (d[both] + reverse[both])

    only_forward = usable_directed & ~usable_directed.T
    only_backward = usable_directed.T & ~usable_directed
    symmetric_distance[only_forward] = d[only_forward]
    symmetric_distance[only_backward] = reverse[only_backward]

    values = symmetric_distance[symmetric_valid]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)

    if values.size == 1:
        ranks[order] = 0.0
    else:
        ranks[order] = np.arange(values.size, dtype=float) / float(values.size - 1)

    quantile = np.zeros((n, n), dtype=float)
    quantile[symmetric_valid] = ranks

    # Global rank kernel: invariant to positive rescaling of distances.
    affinity = np.zeros((n, n), dtype=float)
    affinity[symmetric_valid] = np.exp(-7.0 * quantile[symmetric_valid] ** 2)
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)

    degree = np.sum(affinity, axis=1)
    safe_degree = np.maximum(degree, np.finfo(float).eps)

    normalized = affinity / np.sqrt(
        safe_degree[:, None] * safe_degree[None, :]
    )

    # Effective resistance of the affinity graph.
    laplacian = np.diag(degree) - affinity
    laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)

    resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(resistance, 0.0)
    resistance = 0.5 * (resistance + resistance.T)

    valid_resistance = resistance[symmetric_valid]
    maximum_resistance = float(np.max(valid_resistance)) if valid_resistance.size else 0.0

    if maximum_resistance > 0.0:
        structural_criticality = np.sqrt(resistance / maximum_resistance)
    else:
        structural_criticality = np.zeros((n, n), dtype=float)

    # Combine shortness rank with graph-level structural importance.
    score = normalized * (0.35 + 0.65 * structural_criticality)
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
