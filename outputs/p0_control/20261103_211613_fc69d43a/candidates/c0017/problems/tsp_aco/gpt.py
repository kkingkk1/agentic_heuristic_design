import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Compute a locally contrastive edge heuristic for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    neighbor_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(work, neighbor_count - 1, axis=1)[:, :neighbor_count]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    inverse_distance = 1.0 / np.maximum(work, np.finfo(float).eps)

    source_contrast = local_scale[:, None] / np.maximum(work, np.finfo(float).eps)
    target_contrast = local_scale[None, :] / np.maximum(work, np.finfo(float).eps)
    contrast = np.sqrt(source_contrast * target_contrast)

    ranks = np.argsort(np.argsort(work, axis=1), axis=1).astype(float)
    rank_score = 1.0 - ranks / max(1, n - 2)

    mutual_support = np.sqrt(rank_score * rank_score.T)

    raw = (
        0.35 * inverse_distance
        + 0.40 * contrast
        + 0.25 * mutual_support
    )
    raw[~np.isfinite(raw)] = 0.0
    np.fill_diagonal(raw, 0.0)

    maximum = np.max(raw)
    if maximum > 0.0:
        raw /= maximum

    return raw
