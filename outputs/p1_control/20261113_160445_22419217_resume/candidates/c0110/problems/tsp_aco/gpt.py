import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promising indicators for a TSP distance matrix."""
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distance_matrix < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = distance_matrix.shape[0]
    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    # Use a symmetric working distance for undirected TSP edges.
    d = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    if positive.size == 0:
        return result

    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive)), eps)

    # Robust local scale from several nearest positive edges.
    positive_d = np.where(d > 0, d, np.inf)
    sorted_d = np.sort(positive_d, axis=1)
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    nearest = sorted_d[:, :k]
    finite_nearest = np.where(np.isfinite(nearest), nearest, np.nan)
    local_scale = np.nanmedian(finite_nearest, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    # Stable endpoint-wise ranks; zero-length edges remain highly preferred.
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks.fill(n)
    rank_values = np.arange(1, n + 1, dtype=float)

    for i in range(n):
        ranks[i, order[i]] = rank_values

    np.fill_diagonal(ranks, np.inf)

    scale_i = local_scale[:, None]
    scale_j = local_scale[None, :]

    # Geometric mean of endpoint-normalized distances.
    normalized_i = d / (scale_i + eps)
    normalized_j = d / (scale_j + eps)
    local_cost = np.sqrt(
        np.maximum(normalized_i, 0.0) * np.maximum(normalized_j, 0.0)
    )
    local_factor = np.exp(-0.9 * np.minimum(local_cost, 60.0))

    # Mutual rank: high only when both endpoints rank the edge highly.
    rank_sum = ranks + ranks.T
    mutual_rank = 2.0 / np.maximum(rank_sum, 1.0)
    rank_factor = 0.35 + 1.55 * np.sqrt(np.clip(mutual_rank, 0.0, 1.0))

    # Mild preference for edges connecting similarly dense neighborhoods.
    scale_ratio = (scale_i + eps) / (scale_j + eps)
    balance = np.exp(-0.6 * np.abs(np.log(scale_ratio)))
    balance_factor = 0.7 + 0.5 * balance

    # Global suppression prevents isolated very long edges from dominating.
    global_factor = 0.55 + 0.65 * np.exp(
        -0.7 * np.minimum(d / global_scale, 60.0)
    )

    score = local_factor * rank_factor * balance_factor * global_factor
    score = 0.5 * (score + score.T)

    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
