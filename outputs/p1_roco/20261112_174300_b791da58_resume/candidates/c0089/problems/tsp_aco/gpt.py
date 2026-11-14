import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally calibrated, approximately doubly stochastic TSP indicators."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.max(finite_values))

    scale = max(scale, np.finfo(float).eps)

    # A global heavy-tailed kernel preserves exploration while favoring short edges.
    normalized = np.divide(
        d,
        scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    kernel = 1.0 / np.power(1.0 + np.clip(normalized, 0.0, 1e12), 1.5)
    kernel[~valid] = 0.0

    # Iterative proportional fitting balances source and destination pressure.
    # This creates competition-aware weights without endpoint-local heuristics.
    weights = kernel.copy()
    for _ in range(80):
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = np.divide(
            weights,
            row_sums,
            out=np.zeros_like(weights),
            where=row_sums > 0.0,
        )

        col_sums = weights.sum(axis=0, keepdims=True)
        weights = np.divide(
            weights,
            col_sums,
            out=np.zeros_like(weights),
            where=col_sums > 0.0,
        )

    # Ensure every usable source provides a valid stochastic distribution.
    row_sums = weights.sum(axis=1, keepdims=True)
    result = np.divide(
        weights,
        row_sums,
        out=np.zeros_like(weights),
        where=row_sums > 0.0,
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
