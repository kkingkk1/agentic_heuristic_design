import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return lightweight symmetric edge-priority indicators for stochastic TSP sampling."""
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

    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(positive))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    fallback = float(np.percentile(positive, 85.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = scale

    d = np.where(valid, distances, fallback)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    candidate_count = max(
        1,
        min(n - 1, int(np.ceil(0.85 * np.sqrt(n)))),
    )

    nearest = np.partition(
        d,
        kth=candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    local_scale = np.mean(nearest, axis=1)
    floor = max(scale * 1e-10, np.finfo(float).eps)
    local_scale = np.maximum(local_scale, floor)

    nearest_indices = np.argpartition(
        d,
        kth=candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate_mask[rows, nearest_indices] = True
    candidate_mask |= candidate_mask.T
    candidate_mask &= symmetric_valid

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = d / (pair_scale + floor)
    distance_affinity = 1.0 / (1.0 + normalized_distance)

    endpoint_i = 1.0 / (
        1.0 + d / (local_scale[:, None] + floor)
    )
    endpoint_j = 1.0 / (
        1.0 + d / (local_scale[None, :] + floor)
    )
    endpoint_compatibility = np.sqrt(endpoint_i * endpoint_j)

    candidate_support = np.where(
        candidate_mask,
        1.0,
        0.35,
    )

    score = (
        distance_affinity
        * endpoint_compatibility
        * candidate_support
    )

    score = 0.5 * (score + score.T)
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
