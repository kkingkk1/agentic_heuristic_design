import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    forward_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    reverse_valid = forward_valid.T
    valid = forward_valid | reverse_valid

    if not np.any(valid):
        return result

    forward = np.where(forward_valid, distances, np.inf)
    reverse = np.where(reverse_valid, distances.T, np.inf)

    both = forward_valid & reverse_valid
    only_forward = forward_valid & ~reverse_valid
    only_reverse = reverse_valid & ~forward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both] = 0.5 * (forward[both] + reverse[both])
    distance[only_forward] = forward[only_forward]
    distance[only_reverse] = reverse[only_reverse]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance) & ~diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite = distance[valid]
    positive = finite[finite > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.median(finite))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    # Bounded reciprocal similarity avoids the exponential distance kernels
    # used by the parent heuristics.
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = 1.0 / (
        1.0 + distance[valid] / scale
    )

    # Symmetric Sinkhorn-style balancing reduces domination by popular nodes.
    balanced = affinity.copy()
    for _ in range(24):
        row_sum = balanced.sum(axis=1)
        factor = np.ones(n, dtype=float)
        active = row_sum > eps
        factor[active] = 1.0 / np.sqrt(row_sum[active])
        balanced = factor[:, None] * balanced * factor[None, :]
        balanced[~valid] = 0.0
        balanced = 0.5 * (balanced + balanced.T)

    degree = balanced.sum(axis=1)
    degree_scale = np.maximum(degree, eps)

    # Effective resistance captures structural bridge value rather than only
    # local distance or rank.
    laplacian = np.diag(degree) - balanced
    laplacian = 0.5 * (laplacian + laplacian.T)

    try:
        laplacian_pinv = np.linalg.pinv(
            laplacian,
            hermitian=True,
        )
    except np.linalg.LinAlgError:
        laplacian_pinv = np.linalg.pinv(laplacian)

    resistance = (
        np.diag(laplacian_pinv)[:, None]
        + np.diag(laplacian_pinv)[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = np.maximum(
        np.nan_to_num(resistance, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    resistance[~valid] = 0.0

    resistance_values = resistance[valid]
    resistance_reference = (
        float(np.median(resistance_values))
        if resistance_values.size
        else 1.0
    )
    resistance_reference = max(resistance_reference, eps)

    bridge_signal = np.clip(
        resistance / resistance_reference,
        0.0,
        4.0,
    )

    # Weighted common-neighbor mass estimates how redundant an edge is.
    common_mass = balanced @ balanced
    endpoint_mass = np.sqrt(
        np.maximum(degree[:, None] * degree[None, :], eps)
    )
    redundancy = common_mass / endpoint_mass
    redundancy = np.clip(
        np.nan_to_num(redundancy, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )

    # Prefer edges whose two endpoints are not already saturated.
    degree_reference = max(float(np.median(degree[degree > eps])), eps)
    endpoint_capacity = np.sqrt(
        np.clip(
            degree_reference / degree_scale[:, None],
            0.25,
            4.0,
        )
        * np.clip(
            degree_reference / degree_scale[None, :],
            0.25,
            4.0,
        )
    )

    # A mild length term preserves the actual optimization objective while
    # leaving structural signals responsible for most of the ranking.
    length_term = 1.0 / (
        1.0 + distance / scale
    )
    length_term[~valid] = 0.0

    scores = (
        balanced
        * (0.55 + 0.45 * length_term)
        * (0.72 + 0.28 * np.sqrt(np.clip(bridge_signal, 0.0, 1.0)))
        * (1.0 + 0.18 * np.clip(bridge_signal, 0.0, 4.0))
        * (1.0 - 0.24 * redundancy)
        * np.sqrt(np.maximum(endpoint_capacity, 0.0))
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
