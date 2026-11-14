import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced stochastic edge indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    values = distances[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        lower = float(np.percentile(positive, 25.0))
        upper = float(np.percentile(positive, 75.0))
        median = float(np.median(positive))
    else:
        lower = upper = median = 1.0

    spread = max(upper - lower, eps)
    temperature = max(
        median,
        0.35 * spread,
        eps,
    )

    centered = np.maximum(distances, 0.0) / temperature
    centered = np.clip(centered, 0.0, 80.0)

    kernel = np.exp(-centered)
    kernel[~valid] = 0.0
    np.fill_diagonal(kernel, 0.0)

    positive_kernel = kernel[valid]
    kernel_floor = max(float(np.percentile(positive_kernel, 5.0)), eps)

    kernel = np.where(valid, np.maximum(kernel, kernel_floor), 0.0)

    row_mass = kernel.sum(axis=1)
    col_mass = kernel.sum(axis=0)

    row_mass = np.maximum(row_mass, eps)
    col_mass = np.maximum(col_mass, eps)

    row_factor = 1.0 / np.sqrt(row_mass)
    col_factor = 1.0 / np.sqrt(col_mass)

    balanced = kernel * row_factor[:, None] * col_factor[None, :]

    for _ in range(18):
        row_totals = balanced.sum(axis=1)
        row_totals = np.maximum(row_totals, eps)
        balanced *= (1.0 / row_totals)[:, None]

        col_totals = balanced.sum(axis=0)
        col_totals = np.maximum(col_totals, eps)
        balanced *= (1.0 / col_totals)[None, :]

        balanced = np.where(valid, balanced, 0.0)

    row_totals = balanced.sum(axis=1, keepdims=True)
    row_totals = np.maximum(row_totals, eps)

    row_probability = balanced / row_totals

    column_totals = row_probability.sum(axis=0, keepdims=True)
    column_totals = np.maximum(column_totals, eps)

    scores = row_probability / np.sqrt(column_totals)

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
