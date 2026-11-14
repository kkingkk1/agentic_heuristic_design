import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a globally balanced edge-priority matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_d = np.zeros((n, n), dtype=float)
    symmetric_d[valid] = 0.5 * (d[valid] + d.T[valid])

    positive_values = symmetric_d[valid & (symmetric_d > 0.0)]
    if positive_values.size:
        temperature = float(np.median(positive_values))
    else:
        temperature = 1.0

    temperature = max(temperature, eps)

    active = np.any(valid, axis=1)
    active_rows = np.flatnonzero(active)

    kernel = np.zeros((n, n), dtype=float)

    scaled_distance = symmetric_d / temperature
    scaled_distance = np.clip(scaled_distance, 0.0, 745.0)

    kernel[valid] = np.exp(-scaled_distance[valid])

    # Symmetric entropic balancing:
    # repeatedly rescale both endpoints so edge mass is globally shared
    # instead of being concentrated around already dense nodes.
    scale = np.ones(n, dtype=float)

    for _ in range(300):
        row_mass = scale * (kernel @ scale)
        active_mass = row_mass[active_rows]

        if active_mass.size == 0:
            return np.zeros((n, n), dtype=float)

        active_mass = np.maximum(active_mass, eps)
        correction = np.ones(n, dtype=float)
        correction[active_rows] = 1.0 / np.sqrt(active_mass)

        new_scale = scale * correction
        new_scale = np.clip(new_scale, 1e-150, 1e150)

        if np.max(np.abs(np.log(new_scale[active_rows] / scale[active_rows]))) < 1e-10:
            scale = new_scale
            break

        scale = new_scale

    score = scale[:, None] * kernel * scale[None, :]
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
