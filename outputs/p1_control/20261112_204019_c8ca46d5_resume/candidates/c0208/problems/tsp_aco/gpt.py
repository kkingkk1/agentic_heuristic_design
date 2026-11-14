import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robustly balanced edge priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)

    finite_input = np.isfinite(distances)
    positive_input = finite_input & (distances > 0.0) & ~diagonal

    if not np.any(positive_input):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[positive_input]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(np.isfinite(distances), distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, fallback)

    off_diagonal_values = d[~diagonal]
    global_median = float(np.median(off_diagonal_values))
    global_median = max(global_median, np.finfo(float).eps)

    row_values = d[~diagonal].reshape(n, n - 1)
    row_median = np.median(row_values, axis=1)

    row_deviation = np.abs(row_values - row_median[:, None])
    row_mad = np.median(row_deviation, axis=1)

    eps = np.finfo(float).eps
    robust_scale = 1.4826 * row_mad + 0.25 * global_median
    robust_scale = np.maximum(robust_scale, eps)

    centered_i = (d - row_median[:, None]) / robust_scale[:, None]
    centered_j = (d - row_median[None, :]) / robust_scale[None, :]

    centered_i = np.clip(centered_i, -30.0, 30.0)
    centered_j = np.clip(centered_j, -30.0, 30.0)

    directional_i = 1.0 / (1.0 + np.exp(centered_i))
    directional_j = 1.0 / (1.0 + np.exp(centered_j))

    score = np.sqrt(
        np.maximum(directional_i, 0.0)
        * np.maximum(directional_j, 0.0)
    )
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if not np.any(score > 0.0):
        return np.zeros((n, n), dtype=float)

    scale = np.ones(n, dtype=float)
    target_degree = 1.0

    for _ in range(12):
        row_mass = score @ scale
        update = np.sqrt(
            target_degree / np.maximum(row_mass, eps)
        )
        update = np.clip(update, 1e-6, 1e6)
        scale *= update
        scale = np.clip(scale, 1e-6, 1e6)

    score *= scale[:, None]
    score *= scale[None, :]
    score = 0.5 * (score + score.T)
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
