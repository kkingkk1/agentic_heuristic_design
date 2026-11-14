import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric diffusion-based edge-priority matrix for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    positive = distances[finite & (distances > 0.0)]

    fallback = float(np.max(positive)) if positive.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    d = np.where(finite, distances, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    off_values = d[off_diagonal]

    positive_off = off_values[off_values > 0.0]
    if positive_off.size:
        scale = float(np.median(positive_off))
    else:
        scale = fallback
    scale = max(scale, np.finfo(float).eps)

    normalized = d / scale
    affinity = np.exp(-np.clip(normalized * normalized, 0.0, 50.0))
    affinity[~off_diagonal] = 0.0

    row_mass = np.sum(affinity, axis=1)
    mass_scale = np.maximum(np.mean(row_mass), np.finfo(float).eps)
    degree_factor = np.sqrt(
        np.maximum(row_mass[:, None], np.finfo(float).eps)
        * np.maximum(row_mass[None, :], np.finfo(float).eps)
    ) / mass_scale

    affinity = affinity / np.maximum(degree_factor, np.finfo(float).eps)
    affinity[~off_diagonal] = 0.0

    for _ in range(32):
        row_sum = np.sum(affinity, axis=1)
        affinity = affinity / np.maximum(
            np.sqrt(row_sum[:, None] * row_sum[None, :]),
            np.finfo(float).eps,
        )
        affinity[~off_diagonal] = 0.0

    diffusion = affinity @ affinity
    diffusion = 0.5 * (diffusion + diffusion.T)
    np.fill_diagonal(diffusion, 0.0)

    diffusion_values = diffusion[off_diagonal]
    if diffusion_values.size:
        closure_scale = float(np.percentile(diffusion_values, 75.0))
    else:
        closure_scale = 1.0
    closure_scale = max(closure_scale, np.finfo(float).eps)

    closure = diffusion / closure_scale
    closure = np.clip(closure, 0.0, 1.0)

    score = affinity * (0.82 + 0.18 * closure)
    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
