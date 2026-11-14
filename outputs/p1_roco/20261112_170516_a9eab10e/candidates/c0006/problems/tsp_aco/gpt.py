import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-selection indicators for stochastic TSP solution sampling.

    Higher values indicate edges that are globally short, locally competitive
    for both endpoints, and mutually nearest-neighbor-like.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0)

    if not np.any(positive):
        return np.zeros_like(distances, dtype=float)

    valid_values = distances[positive]
    scale = float(np.median(valid_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.mean(valid_values))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_distances = np.where(
        finite & (distances >= 0),
        distances,
        np.inf,
    )
    safe_distances = np.maximum(safe_distances, 0.0)

    np.fill_diagonal(safe_distances, np.inf)

    global_score = np.exp(-safe_distances / scale)
    global_score[~np.isfinite(global_score)] = 0.0

    row_min = np.min(
        np.where(np.isfinite(safe_distances), safe_distances, np.inf),
        axis=1,
    )
    row_min[~np.isfinite(row_min)] = scale

    relative_distance = safe_distances / (row_min[:, None] + scale * 1e-12)
    local_score = np.exp(-np.maximum(relative_distance - 1.0, 0.0))
    local_score[~np.isfinite(local_score)] = 0.0

    nearest = np.argmin(safe_distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = int(nearest[i])
        if j != i and 0 <= j < n and int(nearest[j]) == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    score = (
        0.50 * global_score
        + 0.35 * np.sqrt(local_score * local_score.T)
        + 0.15 * mutual
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
