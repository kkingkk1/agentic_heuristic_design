import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promising indicators for a TSP distance matrix."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distance_matrix < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros_like(distance_matrix, dtype=float)

    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)

    positive_edges = d[np.isfinite(d) & (d > 0)]
    if positive_edges.size == 0:
        return np.zeros_like(distance_matrix, dtype=float)

    global_scale = max(
        float(np.median(positive_edges)),
        np.finfo(float).eps,
    )

    # Estimate each node's scale from its nearest positive incident edges.
    positive_local = np.where(d > 0, d, np.inf)
    sorted_positive = np.sort(positive_local, axis=1)
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    local_scale = np.median(sorted_positive[:, :k], axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    # Rank edges independently from the perspective of each endpoint.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)

    for i in range(n):
        ranks[i, order[i]] = rank_values

    ranks = np.minimum(ranks, float(n - 1))
    np.fill_diagonal(ranks, np.inf)

    i_scale = local_scale[:, None]
    j_scale = local_scale[None, :]
    scale_sum = i_scale + j_scale

    # Both endpoints must regard the edge as locally compatible.
    local_compatibility = np.exp(
        -2.0 * d / (scale_sum + global_scale * 1e-12)
    )

    # Stronger mutual-rank contribution rewards agreement between endpoints.
    mutual_rank = 1.0 / (ranks + ranks.T)
    rank_factor = 0.45 + 1.35 * mutual_rank

    # Suppress globally long edges without eliminating local structure.
    global_compatibility = np.exp(-0.85 * d / global_scale)
    global_factor = 0.45 + 0.75 * global_compatibility

    # Prefer edges joining nodes with comparable neighborhood scales.
    scale_balance = np.exp(
        -0.75
        * np.abs(
            np.log((i_scale + np.finfo(float).eps)
                   / (j_scale + np.finfo(float).eps))
        )
    )
    balance_factor = 0.55 + 0.65 * scale_balance

    score = (
        local_compatibility
        * rank_factor
        * global_factor
        * balance_factor
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
