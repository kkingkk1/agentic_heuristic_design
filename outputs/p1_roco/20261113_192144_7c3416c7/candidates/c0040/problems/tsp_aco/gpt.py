import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return entropy-balanced edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = d[valid]
    low = float(np.min(values))
    high = float(np.max(values))

    if not np.isfinite(low) or not np.isfinite(high):
        return np.zeros((n, n), dtype=float)

    spread = max(high - low, np.finfo(float).eps)
    normalized = np.zeros((n, n), dtype=float)
    normalized[valid] = (d[valid] - low) / spread

    finite_row = np.where(valid, normalized, np.nan)
    row_center = np.nanmedian(finite_row, axis=1)
    row_center = np.nan_to_num(row_center, nan=0.5)

    centered = np.zeros((n, n), dtype=float)
    centered[valid] = normalized[valid] - row_center[:, None][valid]

    global_quantile = np.zeros((n, n), dtype=float)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    global_quantile[valid] = ranks / max(values.size - 1, 1)

    local_deviation = np.abs(centered)
    temperature = float(np.median(local_deviation[valid]))
    temperature = max(temperature, 0.08)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -global_quantile[valid] / temperature
        -0.35 * np.abs(centered[valid]) / temperature
    )

    row_total = affinity.sum(axis=1)
    alternative = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size == 0:
            continue

        weights = np.sort(affinity[i, neighbors])[::-1]
        strongest = weights[0]
        second = weights[1] if weights.size > 1 else strongest
        baseline = max(second, np.finfo(float).eps)

        alternative[i, neighbors] = affinity[i, neighbors] / baseline
        alternative[i, neighbors] = np.minimum(alternative[i, neighbors], 2.0)

    exchange_pressure = np.sqrt(
        np.maximum(alternative, 0.0)
        * np.maximum(alternative.T, 0.0)
    )

    kernel = affinity * (0.75 + 0.25 * exchange_pressure)
    kernel = 0.5 * (kernel + kernel.T)
    kernel[~valid] = 0.0
    np.fill_diagonal(kernel, 0.0)

    scaling = np.ones(n, dtype=float)
    eps = np.finfo(float).eps

    for _ in range(80):
        row_sum = kernel @ scaling
        updated = 1.0 / np.sqrt(np.maximum(row_sum, eps))
        updated /= max(float(np.mean(updated)), eps)

        if np.max(np.abs(updated - scaling)) < 1e-8:
            scaling = updated
            break

        scaling = 0.65 * scaling + 0.35 * updated

    balanced = kernel * scaling[:, None] * scaling[None, :]
    balanced = 0.5 * (balanced + balanced.T)
    balanced[~valid] = 0.0
    np.fill_diagonal(balanced, 0.0)

    maximum = float(np.max(balanced))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    return (balanced / maximum).astype(float, copy=False)
