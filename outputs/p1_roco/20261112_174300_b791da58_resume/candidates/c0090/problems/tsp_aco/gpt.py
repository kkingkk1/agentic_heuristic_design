import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally calibrated, approximately load-balanced edge weights."""
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

    finite_distances = distances[valid]
    ordered = np.sort(finite_distances)

    positions = np.searchsorted(
        ordered,
        finite_distances,
        side="right",
    )
    quantiles = (positions.astype(float) - 0.5) / max(1.0, float(ordered.size))
    quantiles = np.clip(quantiles, 0.0, 1.0)

    base = np.zeros_like(distances, dtype=float)
    base[valid] = np.exp(-7.0 * quantiles)

    row_counts = valid.sum(axis=1).astype(float)
    col_counts = valid.sum(axis=0).astype(float)

    active_rows = row_counts > 0
    active_cols = col_counts > 0

    if not np.any(active_rows) or not np.any(active_cols):
        return result

    weights = base.copy()
    eps = np.finfo(float).eps

    for _ in range(16):
        row_totals = weights.sum(axis=1, keepdims=True)
        row_scale = np.divide(
            1.0,
            row_totals,
            out=np.zeros_like(row_totals),
            where=row_totals > 0,
        )
        weights *= row_scale

        column_totals = weights.sum(axis=0, keepdims=True)
        desired_columns = active_cols.astype(float)[None, :]
        desired_columns /= max(1.0, float(np.count_nonzero(active_cols)))

        column_scale = np.divide(
            desired_columns,
            np.maximum(column_totals, eps),
            out=np.zeros_like(column_totals),
            where=column_totals > 0,
        )
        weights *= column_scale
        weights[~valid] = 0.0

    row_totals = weights.sum(axis=1, keepdims=True)
    result = np.divide(
        weights,
        row_totals,
        out=np.zeros_like(weights),
        where=row_totals > 0,
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
