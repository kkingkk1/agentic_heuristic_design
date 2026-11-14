import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = distances[valid]
    fallback = float(np.max(positive))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.where(valid, distances, fallback)
    clean = 0.5 * (clean + clean.T)
    valid = np.isfinite(clean) & (clean > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    clean = np.where(valid, clean, fallback)
    np.fill_diagonal(clean, np.inf)

    eps = np.finfo(float).eps
    k = min(4, n - 1)

    nearest_idx = np.argpartition(clean, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]

    nearest_distances = clean[rows, nearest_idx]
    local_scale = np.mean(nearest_distances, axis=1)
    local_scale = np.maximum(local_scale, eps)

    nearest_mask = np.zeros((n, n), dtype=float)
    nearest_mask[rows, nearest_idx] = 1.0
    nearest_mask *= valid

    mutual_support = nearest_mask * nearest_mask.T
    local_support = 0.5 * (nearest_mask + nearest_mask.T)

    relative_distance = clean / (
        np.sqrt(local_scale[:, None] * local_scale[None, :]) + eps
    )
    relative_distance = np.clip(relative_distance, 0.0, 745.0)

    reciprocal = 1.0 / np.maximum(clean, eps)
    reciprocal[~valid] = 0.0

    row_max = np.max(reciprocal, axis=1, keepdims=True)
    reciprocal = np.divide(
        reciprocal,
        row_max,
        out=np.zeros_like(reciprocal),
        where=row_max > 0,
    )

    reciprocal_pair = np.sqrt(reciprocal * reciprocal.T)
    accessibility = np.exp(-relative_distance)
    accessibility[~valid] = 0.0

    score = (
        0.50 * reciprocal_pair
        + 0.30 * accessibility
        + 0.14 * local_support
        + 0.06 * mutual_support
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
