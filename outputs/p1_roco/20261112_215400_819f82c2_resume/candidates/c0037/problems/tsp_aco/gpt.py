import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge-priority indicators for stochastic TSP sampling."""
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
    positive = values[values > 0.0]

    if positive.size:
        temperature = float(np.median(positive))
    else:
        temperature = float(np.median(values))

    if not np.isfinite(temperature) or temperature <= 0.0:
        temperature = 1.0

    temperature = max(temperature, np.finfo(float).eps)

    finite_d = np.where(valid, d, np.inf)
    kernel = np.zeros((n, n), dtype=float)

    scaled = np.clip(finite_d / temperature, 0.0, 700.0)
    kernel[valid] = np.exp(-scaled[valid])

    if not np.any(kernel > 0.0):
        return np.zeros((n, n), dtype=float)

    # Entropic global balancing: equalize outgoing and incoming edge mass.
    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    for _ in range(200):
        row_mass = kernel @ col_scale
        row_scale = np.divide(
            1.0,
            row_mass,
            out=np.zeros_like(row_mass),
            where=row_mass > 0.0,
        )

        col_mass = kernel.T @ row_scale
        col_scale = np.divide(
            1.0,
            col_mass,
            out=np.zeros_like(col_mass),
            where=col_mass > 0.0,
        )

        if np.max(np.abs(row_mass[row_mass > 0.0] - 1.0)) < 1e-10:
            break

    balanced = row_scale[:, None] * kernel * col_scale[None, :]
    balanced[~valid] = 0.0

    # Remove directional artifacts while preserving globally balanced evidence.
    score = 0.5 * (balanced + balanced.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
