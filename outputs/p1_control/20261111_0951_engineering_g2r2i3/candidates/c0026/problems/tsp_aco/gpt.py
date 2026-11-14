import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")

    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError("distance_matrix must be square")

    n = n_rows
    if n <= 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = distance_matrix.copy()
    finite_positive = np.isfinite(d) & (d > 0)

    if not np.any(finite_positive):
        return np.zeros_like(d, dtype=float)

    positive_values = d[finite_positive]
    scale = float(np.median(positive_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(positive_values))

    invalid_value = max(scale, np.finfo(float).eps) * 10.0
    d[~np.isfinite(d)] = invalid_value
    d[d <= 0] = invalid_value
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps

    # Reciprocal local distance preference from each endpoint.
    inv_d = 1.0 / np.maximum(d, eps)
    inv_d /= np.maximum(
        np.max(inv_d, axis=1, keepdims=True),
        eps,
    )

    # Reciprocal local rank preference from each endpoint.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        eps,
    )

    # Require an edge to be reasonably attractive from both endpoints.
    endpoint_preference = 0.65 * inv_d + 0.35 * rank_score
    reciprocal_preference = np.sqrt(
        endpoint_preference * endpoint_preference.T
    )

    # Mild structural preservation: favor edges connecting locally flexible
    # endpoints, without imposing a strict mutual-nearest-neighbor rule.
    local_scale = np.partition(d, 1, axis=1)[:, 1]
    local_scale = np.maximum(local_scale, eps)
    structural = np.minimum(
        local_scale[:, None] / np.maximum(d, eps),
        local_scale[None, :] / np.maximum(d, eps),
    )
    structural = np.clip(structural, 0.0, 1.0)

    heuristic = (
        0.78 * reciprocal_preference
        + 0.22 * structural
    )

    heuristic = np.nan_to_num(
        heuristic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    heuristic = np.maximum(heuristic, 0.0)
    np.fill_diagonal(heuristic, 0.0)

    return heuristic
