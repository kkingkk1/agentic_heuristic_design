import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return lightweight symmetric candidate-edge indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    symmetric_valid = valid & valid.T
    positive = distances[valid & (distances > 0.0)]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(symmetric_valid, distances, fallback)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive_scale = float(np.median(positive))
    scale_floor = max(
        positive_scale * 1e-10,
        np.finfo(float).eps,
    )

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))

    candidate_indices = np.argpartition(
        d,
        kth=k - 1,
        axis=1,
    )[:, :k]

    row_indices = np.arange(n)[:, None]
    local_values = d[row_indices, candidate_indices]
    local_scale = np.mean(local_values, axis=1)
    local_scale = np.maximum(local_scale, scale_floor)

    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_mask[row_indices, candidate_indices] = True
    candidate_mask &= symmetric_valid
    candidate_mask |= candidate_mask.T
    np.fill_diagonal(candidate_mask, False)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = d / (pair_scale + scale_floor)
    distance_score = 1.0 / (1.0 + normalized_distance)

    alternative = np.partition(d, kth=min(1, n - 2), axis=1)[
        :, min(1, n - 2)
    ]
    alternative = np.maximum(alternative, scale_floor)

    alternative_scale = np.sqrt(
        alternative[:, None] * alternative[None, :]
    )
    neighborhood_bonus = 1.0 + 0.08 * (
        alternative_scale / (d + alternative_scale + scale_floor)
    )

    score = distance_score * neighborhood_bonus
    score = 0.5 * (score + score.T)
    score[~candidate_mask] = 0.0
    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
