import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge indicators using global-kernel balancing."""
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

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = float(np.max(finite_values))

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    safe_distances = np.where(valid, distances, 0.0)

    normalized = np.divide(
        safe_distances,
        scale,
        out=np.zeros_like(safe_distances, dtype=float),
        where=valid,
    )
    normalized = np.clip(normalized, 0.0, 1.0e6)

    # Global heavy-tailed kernel: retain exploration for moderately long edges.
    affinity = 1.0 / (1.0 + normalized * normalized)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    # Iterative row/column balancing discourages both source and target hubs.
    balanced = affinity.copy()

    for _ in range(32):
        row_totals = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_totals,
            out=np.zeros_like(balanced),
            where=row_totals > 0.0,
        )

        column_totals = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            column_totals,
            out=np.zeros_like(balanced),
            where=column_totals > 0.0,
        )

        balanced[~valid] = 0.0

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
    ).astype(float, copy=False)
