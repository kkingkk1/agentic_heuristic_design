import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced stochastic edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    finite_values = distances[valid]
    scale = float(np.median(finite_values))
    if not np.isfinite(scale) or scale <= 0.0:
        positive = finite_values[finite_values > 0.0]
        scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    normalized = np.divide(
        distances,
        scale,
        out=np.zeros_like(distances, dtype=float),
        where=valid,
    )

    kernel = np.zeros_like(distances, dtype=float)
    kernel[valid] = np.exp(-np.clip(normalized[valid] / 0.85, 0.0, 50.0))

    # Iterative proportional fitting creates a globally balanced edge measure.
    balanced = kernel.copy()
    for _ in range(48):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=col_sum > 0.0,
        )

    # Sampling requires each source row to define a probability distribution.
    row_sum = balanced.sum(axis=1, keepdims=True)
    result = np.divide(
        balanced,
        row_sum,
        out=np.zeros_like(balanced),
        where=row_sum > 0.0,
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
