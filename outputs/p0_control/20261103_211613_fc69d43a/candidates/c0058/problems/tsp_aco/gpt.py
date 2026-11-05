import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distance_matrix < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    distance = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(distance, np.inf)

    positive = distance[(distance > 0) & np.isfinite(distance)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    k = min(4, n - 1)
    nearest = np.partition(distance, k - 1, axis=1)[:, :k]
    local_radius = np.maximum(nearest.mean(axis=1), eps)

    radius_product = local_radius[:, None] * local_radius[None, :]
    pair_scale = np.sqrt(radius_product)

    relative_length = distance / np.maximum(pair_scale, eps)

    density = 1.0 / np.sqrt(np.maximum(radius_product, eps))
    finite_density = density[np.isfinite(density)]
    if finite_density.size:
        density /= np.max(finite_density)

    score = np.exp(-relative_length) * (0.5 + 0.5 * density)
    score[~np.isfinite(score)] = 0.0

    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
