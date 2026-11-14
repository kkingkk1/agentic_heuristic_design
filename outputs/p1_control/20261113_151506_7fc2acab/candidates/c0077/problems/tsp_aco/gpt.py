import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    d = np.maximum(distances, 0.0).copy()
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps
    base_scale = max(base_scale, eps)

    kth = min(2, n - 2)
    local_scale = np.partition(d, kth, axis=1)[:, kth]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        base_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    closeness = np.exp(-d / pair_scale)
    closeness[~np.isfinite(closeness)] = 0.0

    mutual_closeness = np.sqrt(closeness * closeness.T)
    directional_average = 0.5 * (closeness + closeness.T)

    score = 0.8 * mutual_closeness + 0.2 * directional_average
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1)
    pair_max = np.sqrt(row_max[:, None] * row_max[None, :])

    score = np.divide(
        score,
        pair_max,
        out=np.zeros_like(score),
        where=pair_max > 0.0,
    )
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
