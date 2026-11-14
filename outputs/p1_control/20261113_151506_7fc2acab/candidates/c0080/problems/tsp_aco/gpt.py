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
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    local_scale = np.partition(d, min(2, n - 2), axis=1)[:, min(2, n - 2)]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        scale,
    )
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

    closeness = pair_scale / (pair_scale + d)
    closeness[~np.isfinite(closeness)] = 0.0

    score = np.sqrt(closeness * closeness.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
