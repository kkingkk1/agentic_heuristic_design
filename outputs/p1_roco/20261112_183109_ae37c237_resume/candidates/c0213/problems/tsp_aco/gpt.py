import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally calibrated, degree-balanced edge-priority indicators."""
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
        q25, q50, q75 = np.quantile(
            positive,
            [0.25, 0.50, 0.75],
        )
        scale = float(q50)
        spread = float(q75 - q25)
    else:
        q25 = q50 = q75 = float(np.max(values))
        scale = q50
        spread = 0.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    if not np.isfinite(spread) or spread <= eps:
        spread = scale

    finite_max = float(np.max(values))
    if not np.isfinite(finite_max):
        finite_max = scale

    safe_distance = np.where(valid, distances, finite_max + 8.0 * scale)

    flat_values = values.astype(float, copy=False)
    sorted_values = np.sort(flat_values)

    positions = np.searchsorted(
        sorted_values,
        safe_distance,
        side="right",
    )
    global_percentile = positions / float(max(1, sorted_values.size))

    centered = (safe_distance - scale) / max(spread, eps)

    short_edge_kernel = 1.0 / (
        1.0
        + np.log1p(np.maximum(safe_distance, 0.0) / scale)
    )

    distribution_kernel = np.exp(
        -0.75 * np.maximum(centered, -4.0)
    )
    distribution_kernel = np.clip(distribution_kernel, 0.0, 4.0)

    percentile_kernel = np.power(
        np.clip(1.0 - global_percentile, 0.0, 1.0),
        1.35,
    )

    raw = (
        0.20
        + 0.45 * short_edge_kernel
        + 0.25 * percentile_kernel
        + 0.10 * distribution_kernel
    )

    raw[~valid] = 0.0
    raw = np.nan_to_num(
        raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    active_rows = raw.sum(axis=1) > 0.0
    active_cols = raw.sum(axis=0) > 0.0

    balanced = raw.copy()

    for _ in range(40):
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

        balanced[~valid] = 0.0

    row_sum = balanced.sum(axis=1, keepdims=True)
    scores = np.divide(
        balanced,
        row_sum,
        out=np.zeros_like(balanced),
        where=row_sum > 0.0,
    )

    scores[~active_rows, :] = 0.0
    scores[:, ~active_cols] = 0.0
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return scores.astype(float, copy=False)
