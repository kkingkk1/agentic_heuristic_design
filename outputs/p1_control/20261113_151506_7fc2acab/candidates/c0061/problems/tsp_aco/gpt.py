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
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive_edges)), eps)

    # Use a slightly broader local neighborhood for a smoother scale estimate.
    positive_local = np.where(np.isfinite(d) & (d > 0), d, np.nan)
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = positive_local[i]
        valid = row[np.isfinite(row)]
        if valid.size:
            nearest = np.partition(valid, min(k, valid.size) - 1)[:k]
            local_scale[i] = max(float(np.median(nearest)), eps)

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)

    for i in range(n):
        ranks[i, order[i]] = rank_values

    ranks = np.minimum(ranks, float(n - 1))
    np.fill_diagonal(ranks, np.inf)

    i_scale = local_scale[:, None]
    j_scale = local_scale[None, :]
    pair_scale = i_scale + j_scale

    # Stronger local compatibility emphasizes agreement at both endpoints.
    local_compatibility = np.exp(-1.15 * d / (pair_scale + 0.05 * global_scale))

    # Harmonic-style rank agreement suppresses edges favored by only one endpoint.
    mutual_rank = 2.0 / (ranks + ranks.T)

    # Global compatibility limits the influence of locally irregular long edges.
    global_compatibility = np.exp(-0.85 * d / global_scale)

    # Penalize edges connecting nodes with substantially different local scales.
    scale_balance = np.exp(
        -0.75 * np.abs(np.log((i_scale + eps) / (j_scale + eps)))
    )

    score = (
        local_compatibility
        * (0.45 + 1.15 * mutual_rank)
        * (0.55 + 0.90 * global_compatibility)
        * (0.55 + 0.85 * scale_balance)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
