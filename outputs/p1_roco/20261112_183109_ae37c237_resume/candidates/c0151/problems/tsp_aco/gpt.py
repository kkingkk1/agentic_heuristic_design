import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally calibrated, destination-balanced TSP sampling weights."""
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

    values = distances[valid]
    q10, q90 = np.quantile(values, [0.10, 0.90])
    spread = float(q90 - q10)

    if not np.isfinite(spread) or spread <= np.finfo(float).eps:
        positive = values[values > 0.0]
        scale = float(np.median(positive)) if positive.size else 1.0
        scale = max(scale, np.finfo(float).eps)

        transformed = np.log1p(
            np.divide(
                distances,
                scale,
                out=np.zeros_like(distances),
                where=valid,
            )
        )
        finite_transformed = transformed[valid]
        upper = float(np.max(finite_transformed))
        if not np.isfinite(upper) or upper <= np.finfo(float).eps:
            contrast = np.zeros_like(distances)
        else:
            contrast = transformed / upper
    else:
        contrast = np.clip(
            (distances - q10) / spread,
            0.0,
            1.0,
        )

    # A nonzero floor preserves exploration while favoring globally short edges.
    affinity = np.zeros_like(distances, dtype=float)
    affinity[valid] = 0.025 + np.square(1.0 - contrast[valid])

    # Iterative destination balancing discourages many sources from collapsing
    # onto the same destination without using explicit neighbor ranks.
    weights = affinity.copy()
    eps = np.finfo(float).eps

    for _ in range(5):
        destination_load = weights.sum(axis=0)
        positive_load = destination_load > 0.0
        correction = np.ones(n, dtype=float)
        correction[positive_load] = 1.0 / np.sqrt(
            destination_load[positive_load]
        )

        weights *= correction[None, :]
        row_totals = weights.sum(axis=1, keepdims=True)
        weights = np.divide(
            weights,
            row_totals,
            out=np.zeros_like(weights),
            where=row_totals > eps,
        )

    weights[~valid] = 0.0
    np.fill_diagonal(weights, 0.0)

    row_totals = weights.sum(axis=1, keepdims=True)
    return np.divide(
        weights,
        row_totals,
        out=result,
        where=row_totals > eps,
    )
