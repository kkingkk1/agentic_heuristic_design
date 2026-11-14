import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    d = d.copy()
    np.fill_diagonal(d, np.inf)

    finite_edges = d[np.isfinite(d)]
    positive_edges = finite_edges[finite_edges > 0.0]

    eps = np.finfo(float).eps
    global_scale = float(np.median(positive_edges)) if positive_edges.size else 1.0
    global_scale = max(global_scale, eps)

    nearest = np.min(d, axis=1)
    safe_nearest = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        global_scale,
    )

    finite_nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if finite_nearest.size:
        global_scale = max(float(np.median(finite_nearest)), eps)

    local_scale = np.sqrt(np.outer(safe_nearest, safe_nearest))
    local_scale = np.maximum(local_scale, eps)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized_distance = d / global_scale
        desirability = 1.0 / (1.0 + normalized_distance * normalized_distance)

        local_ratio = local_scale / (d + 0.8 * local_scale)
        local_ratio = np.clip(local_ratio, 0.0, 1.0)

    nearest_index = np.argmin(d, axis=1)
    indices = np.arange(n)

    mutual_nearest = (
        (nearest_index[:, None] == indices[None, :])
        & (nearest_index[None, :] == indices[:, None])
    )
    np.fill_diagonal(mutual_nearest, False)

    scores = desirability * (0.60 + 0.40 * local_ratio)
    scores *= 1.0 + 0.40 * mutual_nearest.astype(float)

    scores = 0.5 * (scores + scores.T)
    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
