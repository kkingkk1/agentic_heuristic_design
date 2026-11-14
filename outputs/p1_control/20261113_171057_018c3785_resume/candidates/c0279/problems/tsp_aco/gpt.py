import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced, regret-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]

    if positive.size:
        scale = float(np.median(positive))
    else:
        finite = d[valid]
        scale = float(np.median(finite)) if finite.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    masked = np.where(valid, d, np.inf)

    row_min = np.min(masked, axis=1)
    col_min = np.min(masked, axis=0)

    row_sorted = np.sort(masked, axis=1)
    col_sorted = np.sort(masked, axis=0)

    row_second = row_sorted[:, 1]
    col_second = col_sorted[1, :]

    row_gap = row_second - row_min
    col_gap = col_second - col_min

    row_gap_values = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    col_gap_values = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

    row_gap_scale = (
        float(np.median(row_gap_values))
        if row_gap_values.size
        else scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values))
        if col_gap_values.size
        else scale
    )

    if not np.isfinite(row_gap_scale) or row_gap_scale <= 0.0:
        row_gap_scale = scale
    if not np.isfinite(col_gap_scale) or col_gap_scale <= 0.0:
        col_gap_scale = scale

    row_delta = np.zeros_like(d)
    col_delta = np.zeros_like(d)

    np.subtract(
        d,
        row_min[:, None],
        out=row_delta,
        where=valid,
    )
    np.subtract(
        d,
        col_min[None, :],
        out=col_delta,
        where=valid,
    )

    row_delta = np.maximum(row_delta, 0.0)
    col_delta = np.maximum(col_delta, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        row_regret = np.exp(
            -np.clip(row_delta / row_gap_scale, 0.0, 60.0)
        )
        col_regret = np.exp(
            -np.clip(col_delta / col_gap_scale, 0.0, 60.0)
        )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0
    regret[~np.isfinite(regret)] = 0.0

    temperature = max(0.38 * scale, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logits = 0.5 * (row_delta + col_delta) / temperature
        affinity = np.exp(-np.clip(logits, 0.0, 60.0))

    affinity[~valid] = 0.0
    affinity[~np.isfinite(affinity)] = 0.0

    balanced = affinity.copy()

    for _ in range(48):
        previous = balanced.copy()

        row_sum = np.sum(balanced, axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(row_sum) & (row_sum > 0.0),
        )

        col_sum = np.sum(balanced, axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=np.isfinite(col_sum) & (col_sum > 0.0),
        )

        balanced[~np.isfinite(balanced)] = 0.0

        if np.max(np.abs(balanced - previous)) <= 1e-10:
            break

    with np.errstate(over="ignore", invalid="ignore"):
        result = balanced * np.power(regret + eps, 0.9)

    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
