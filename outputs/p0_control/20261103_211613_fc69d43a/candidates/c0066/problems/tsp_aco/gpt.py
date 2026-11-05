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

    d = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps * max(scale, 1.0)

    k = max(1, min(6, n - 1))
    nearest = np.partition(d, kth=k - 1, axis=1)[:, :k]
    local_radius = np.mean(nearest, axis=1)
    local_radius = np.maximum(local_radius, eps)

    pair_radius = np.sqrt(
        local_radius[:, None] * local_radius[None, :]
    )
    normalized_length = d / np.maximum(pair_radius, eps)

    density_bonus = 1.0 / np.sqrt(
        np.maximum(local_radius[:, None] * local_radius[None, :], eps)
    )
    finite_density = density_bonus[np.isfinite(density_bonus)]
    if finite_density.size:
        density_bonus /= max(float(np.max(finite_density)), eps)
    else:
        density_bonus.fill(0.0)

    score = np.exp(-1.15 * normalized_length)
    score *= 0.4 + 0.6 * density_bonus

    score[~np.isfinite(score)] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
