import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return prior edge-selection indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n <= 1:
        return score

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    finite_values = d[valid]
    if finite_values.size == 0:
        return score

    fallback = float(np.max(finite_values))
    work = np.where(valid, d, fallback * 2.0 + 1.0)
    np.fill_diagonal(work, np.inf)

    # Scale-free inverse-distance desirability.
    positive = finite_values[finite_values > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    inv_distance = 1.0 / (1.0 + work / max(scale, np.finfo(float).eps))
    inv_distance[~valid] = 0.0

    # Endpoint-specific rank: rank 1 means the nearest valid neighbor.
    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        row_valid = valid[i]
        order = np.argsort(work[i])
        rank = 0
        for j in order:
            if row_valid[j]:
                ranks[i, j] = rank
                rank += 1

    rank_quality = np.exp(-0.7 * (ranks / max(n - 1, 1)))
    rank_quality[~valid] = 0.0

    # Reward edges that are attractive from both endpoints.
    mutual_support = np.sqrt(rank_quality * rank_quality.T)

    # Prefer edges that are competitive at both endpoints.
    local_competitiveness = 0.5 * (rank_quality + rank_quality.T)

    score = (
        0.50 * inv_distance
        + 0.30 * mutual_support
        + 0.20 * local_competitiveness
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return score
