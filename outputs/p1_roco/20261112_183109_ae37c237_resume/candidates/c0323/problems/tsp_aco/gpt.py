import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return an entropically balanced edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros_like(distances, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif finite_values.size:
        scale = float(np.max(finite_values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    relative = np.divide(
        distances,
        scale,
        out=np.full_like(distances, np.inf, dtype=float),
        where=valid,
    )

    log_distance = np.log1p(np.clip(relative, 0.0, 1e300))
    kernel = np.exp(-np.clip(log_distance / 0.72, 0.0, 50.0))
    kernel[~valid] = 0.0
    np.fill_diagonal(kernel, 0.0)

    row_factor = np.ones(n, dtype=float)
    col_factor = np.ones(n, dtype=float)

    for _ in range(24):
        row_mass = kernel @ col_factor
        row_factor = np.divide(
            1.0,
            row_mass,
            out=np.zeros_like(row_mass),
            where=row_mass > eps,
        )

        col_mass = kernel.T @ row_factor
        col_factor = np.divide(
            1.0,
            col_mass,
            out=np.zeros_like(col_mass),
            where=col_mass > eps,
        )

    balanced = (
        row_factor[:, None]
        * kernel
        * col_factor[None, :]
    )

    balanced[~valid] = 0.0
    np.fill_diagonal(balanced, 0.0)
    balanced = np.nan_to_num(
        balanced,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    totals = balanced.sum(axis=1, keepdims=True)

    return np.divide(
        balanced,
        totals,
        out=np.zeros_like(balanced),
        where=totals > 0.0,
    )
