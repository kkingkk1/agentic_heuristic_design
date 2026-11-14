import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    distances = np.array(distance_matrix, dtype=float, copy=True)
    distances[~np.isfinite(distances)] = np.inf
    np.fill_diagonal(distances, np.inf)

    finite_positive = distances[np.isfinite(distances) & (distances > 0)]
    if finite_positive.size == 0:
        return np.zeros_like(distances, dtype=float)

    global_scale = float(np.median(finite_positive))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    row_scales = np.zeros(n, dtype=float)
    for i in range(n):
        row_values = distances[i]
        valid = row_values[np.isfinite(row_values) & (row_values > 0)]
        row_scales[i] = (
            float(np.median(valid)) if valid.size else global_scale
        )

    row_scales = np.maximum(row_scales, np.finfo(float).eps)

    normalized = distances / np.sqrt(row_scales[:, None] * row_scales[None, :])
    normalized[~np.isfinite(normalized)] = np.inf

    exponential_prior = np.zeros((n, n), dtype=float)
    valid = np.isfinite(normalized) & (normalized > 0)
    exponential_prior[valid] = np.exp(-normalized[valid])

    reciprocal_prior = np.zeros((n, n), dtype=float)
    reciprocal_prior[valid] = 1.0 / (1.0 + normalized[valid])

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        candidates = np.where(np.isfinite(distances[i]) & (distances[i] > 0))[0]
        if candidates.size:
            nearest[i] = candidates[np.argmin(distances[i, candidates])]

    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_bonus[i, j] = 1.0

    prior = (
        0.50 * exponential_prior
        + 0.30 * reciprocal_prior
        + 0.20 * mutual_bonus
    )

    prior = 0.5 * (prior + prior.T)
    np.fill_diagonal(prior, 0.0)

    maximum = float(np.max(prior)) if prior.size else 0.0
    if maximum > 0:
        prior /= maximum

    return prior
