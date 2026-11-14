import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a diffusion-based symmetric edge-priority matrix for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    nonnegative = distances >= 0.0
    valid = finite & nonnegative
    np.fill_diagonal(valid, False)

    symmetric_valid = valid & valid.T
    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distance = 0.5 * (
        np.where(symmetric_valid, distances, 0.0)
        + np.where(symmetric_valid, distances.T, 0.0)
    )

    positive = symmetric_distance[symmetric_valid & (symmetric_distance > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    safe_distance = np.where(symmetric_valid, symmetric_distance, 0.0)

    direct_affinity = np.exp(
        -np.clip(safe_distance / scale, 0.0, 60.0)
    )
    direct_affinity *= symmetric_valid
    np.fill_diagonal(direct_affinity, 0.0)

    degree = np.sum(direct_affinity, axis=1)
    degree_floor = np.finfo(float).eps
    safe_degree = np.maximum(degree, degree_floor)

    transition = direct_affinity / safe_degree[:, None]

    row_norm = np.sqrt(
        np.sum(transition * transition, axis=1)
    )
    row_norm = np.maximum(row_norm, degree_floor)

    diffusion_similarity = (
        transition @ transition.T
    ) / (row_norm[:, None] * row_norm[None, :])
    diffusion_similarity = np.clip(diffusion_similarity, 0.0, 1.0)

    spectral_affinity = direct_affinity / np.sqrt(
        safe_degree[:, None] * safe_degree[None, :]
    )
    spectral_max = float(np.max(spectral_affinity))
    if np.isfinite(spectral_max) and spectral_max > 0.0:
        spectral_affinity = spectral_affinity / spectral_max

    score = (
        np.power(np.maximum(direct_affinity, 0.0), 0.65)
        * (0.72 + 0.28 * diffusion_similarity)
        * (0.78 + 0.22 * spectral_affinity)
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
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
