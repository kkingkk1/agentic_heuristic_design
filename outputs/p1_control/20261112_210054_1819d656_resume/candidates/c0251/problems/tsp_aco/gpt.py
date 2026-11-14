import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric endpoint-aware edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    positive = distances[valid & (distances > 0.0)]
    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.percentile(positive, 90.0))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(valid, distances, fallback)
    d = 0.5 * (d + d.T)
    symmetric_valid = valid & valid.T
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive)), eps)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n) + 0.5))))
    local_candidates = np.partition(d, kth=k - 1, axis=1)[:, :k]

    local_scale = np.mean(local_candidates, axis=1)
    local_scale = np.maximum(local_scale, global_scale * 1e-10)

    if k >= 2:
        ordered_local = np.partition(local_candidates, kth=1, axis=1)
        nearest = ordered_local[:, 0]
        alternative = ordered_local[:, 1]
    else:
        nearest = local_candidates[:, 0]
        alternative = local_candidates[:, 0]

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = d / (pair_scale + eps)

    local_affinity = 1.0 / np.square(1.0 + relative_distance)
    global_affinity = 1.0 / (1.0 + d / (global_scale + eps))

    nearest_scale = np.sqrt(
        np.maximum(nearest[:, None], eps)
        * np.maximum(nearest[None, :], eps)
    )
    nearest_support = 1.0 / (1.0 + d / (nearest_scale + eps))

    alternative_scale = np.sqrt(
        np.maximum(alternative[:, None], eps)
        * np.maximum(alternative[None, :], eps)
    )
    margin = alternative_scale / (
        d + alternative_scale + local_scale.min() * 1e-10 + eps
    )
    margin = np.clip(margin, 0.0, 1.0)

    score = local_affinity
    score *= 0.84 + 0.10 * global_affinity + 0.06 * nearest_support
    score *= 1.0 + 0.12 * margin
    score = 0.5 * (score + score.T)

    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
