import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return entropy-balanced stochastic edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps

    normalized = np.divide(
        np.where(valid, distances, 0.0),
        max(scale, eps),
    )

    # A smooth power-law affinity is less sensitive to extreme outliers
    # than an exponential distance kernel.
    affinity = np.power(1.0 + np.maximum(normalized, 0.0), -1.35)

    # Retain a small exploration floor on every valid edge.
    affinity = 0.08 + 0.92 * affinity
    affinity[~valid] = 0.0

    # Approximate a doubly stochastic edge allocation.
    # Invalid rows or columns remain zero throughout the balancing process.
    balanced = affinity.copy()

    for _ in range(48):
        row_totals = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_totals,
            out=np.zeros_like(balanced),
            where=row_totals > 0.0,
        )

        col_totals = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_totals,
            out=np.zeros_like(balanced),
            where=col_totals > 0.0,
        )

        balanced[~valid] = 0.0

    # Restore a row-stochastic sampling distribution.
    row_totals = balanced.sum(axis=1, keepdims=True)
    result = np.divide(
        balanced,
        row_totals,
        out=np.zeros_like(balanced),
        where=row_totals > 0.0,
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
