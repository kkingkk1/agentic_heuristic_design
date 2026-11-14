import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-inclusion priorities for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distances.copy()
    finite_off_diagonal = np.isfinite(d)
    np.fill_diagonal(finite_off_diagonal, False)

    if not np.any(finite_off_diagonal):
        return np.zeros((n, n), dtype=float)

    finite_values = d[finite_off_diagonal]
    fallback = float(np.max(finite_values))
    fallback = max(fallback, 0.0)

    d[~np.isfinite(d)] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive_values = d[np.isfinite(d) & (d > 0.0)]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inverse_distance = 1.0 / (1.0 + d / scale)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    candidate_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    candidate_indices = np.argpartition(
        d,
        kth=candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    candidate_distances = np.take_along_axis(d, candidate_indices, axis=1)
    local_order = np.argsort(candidate_distances, axis=1, kind="stable")
    candidate_indices = np.take_along_axis(
        candidate_indices,
        local_order,
        axis=1,
    )

    rank_score = np.zeros((n, n), dtype=float)
    local_ranks = np.arange(candidate_count, dtype=float)
    local_values = 1.0 / (1.0 + local_ranks)
    rows = np.arange(n)[:, None]
    rank_score[rows, candidate_indices] = local_values[None, :]

    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_mask[rows, candidate_indices] = True
    np.fill_diagonal(candidate_mask, False)

    mutual = candidate_mask & candidate_mask.T
    one_sided = candidate_mask | candidate_mask.T

    score = (
        0.60 * inverse_distance
        + 0.22 * (rank_score + rank_score.T) / 2.0
        + 0.13 * mutual.astype(float)
        + 0.05 * one_sided.astype(float)
    )

    score = (score + score.T) / 2.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
