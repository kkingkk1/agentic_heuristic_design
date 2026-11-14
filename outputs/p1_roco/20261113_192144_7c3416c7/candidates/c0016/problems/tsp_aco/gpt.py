import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Multi-scale neighborhood-consistency heuristic for TSP edge sampling.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Square pairwise distance matrix.

    Returns
    -------
    np.ndarray
        A nonnegative matrix with the same shape as distance_matrix.
        Larger values indicate more promising edges.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances, dtype=float)
    if n == 1:
        return np.zeros_like(distances, dtype=float)

    d = np.array(distances, copy=True)
    d[~np.isfinite(d)] = np.inf
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    positive_values = finite_values[finite_values > 0]
    fallback = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.max(finite_values)), 1.0)
    )
    d[~np.isfinite(d)] = fallback * 10.0

    # Use a robust global scale to avoid domination by distance outliers.
    global_scale = float(np.median(d[np.isfinite(d)]))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = fallback
    global_scale = max(global_scale, np.finfo(float).eps)

    # Rank-based local scale: robust to arbitrary distance magnitudes.
    order = np.argsort(d, axis=1)
    rank_grid = np.arange(n)
    local_rank = np.empty_like(order, dtype=float)
    local_rank[np.arange(n)[:, None], order] = rank_grid[None, :]

    k = max(2, min(n - 1, int(np.ceil(np.sqrt(n)))))
    local_scale = np.take_along_axis(
        d, order[:, :k], axis=1
    ).mean(axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Short-range preference from both absolute and endpoint-relative distance.
    absolute_preference = np.exp(-np.sqrt(np.maximum(d, 0.0) / global_scale))
    endpoint_preference = np.exp(
        -0.5
        * (
            np.sqrt(d / local_scale[:, None])
            + np.sqrt(d / local_scale[None, :])
        )
    )

    # Mutual neighborhood preference: high when both endpoints rank each other well.
    rank_preference = 1.0 / (
        1.0
        + np.sqrt(
            (local_rank + 1.0) * (local_rank.T + 1.0)
        )
    )

    # Prefer edges connecting nodes with compatible local distance scales.
    scale_balance = np.exp(
        -np.abs(np.log(local_scale[:, None] / local_scale[None, :]))
    )

    # A mild common-neighbor signal rewards edges embedded in a dense local region.
    neighborhood_radius = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    common_support = np.exp(
        -np.mean(
            np.minimum(
                d / np.maximum(local_scale[:, None], np.finfo(float).eps),
                d / np.maximum(local_scale[None, :], np.finfo(float).eps),
            ),
            axis=0,
        )
    )
    common_support = np.broadcast_to(common_support, (n, n))

    score = (
        absolute_preference ** 0.30
        * endpoint_preference ** 0.35
        * rank_preference ** 0.20
        * scale_balance ** 0.10
        * (0.85 + 0.15 * common_support)
    )

    # Suppress self-loops and sanitize numerical edge cases.
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    # Row normalization preserves per-node sampling comparability.
    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0,
    )

    return score.astype(float, copy=False)
