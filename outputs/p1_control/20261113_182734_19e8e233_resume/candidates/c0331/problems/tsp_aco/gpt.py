import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced, regret-aware priors for directed TSP edges."""
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
    masked = np.where(valid, d, np.inf)

    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        distance_scale = float(np.median(positive_values))
    else:
        distance_scale = max(float(np.median(valid_values)), 1.0)

    distance_scale = max(distance_scale, eps)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_candidates = np.partition(masked, kth=1, axis=1)[:, :2]
    col_candidates = np.partition(masked, kth=1, axis=0)[:2, :]

    row_best = row_candidates[:, 0]
    row_second = row_candidates[:, 1]
    col_best = col_candidates[0, :]
    col_second = col_candidates[1, :]

    row_best = np.where(row_count > 0, row_best, 0.0)
    col_best = np.where(col_count > 0, col_best, 0.0)

    row_excess = np.zeros_like(d)
    col_excess = np.zeros_like(d)

    np.subtract(
        d,
        row_best[:, None],
        out=row_excess,
        where=valid,
    )
    np.subtract(
        d,
        col_best[None, :],
        out=col_excess,
        where=valid,
    )

    np.maximum(row_excess, 0.0, out=row_excess, where=valid)
    np.maximum(col_excess, 0.0, out=col_excess, where=valid)

    row_gap = np.where(
        row_count >= 2,
        np.maximum(row_second - row_best, 0.0),
        distance_scale,
    )
    col_gap = np.where(
        col_count >= 2,
        np.maximum(col_second - col_best, 0.0),
        distance_scale,
    )

    row_gap_values = row_gap[
        np.isfinite(row_gap) & (row_gap > 0.0)
    ]
    col_gap_values = col_gap[
        np.isfinite(col_gap) & (col_gap > 0.0)
    ]

    row_gap_scale = (
        float(np.median(row_gap_values))
        if row_gap_values.size
        else distance_scale
    )
    col_gap_scale = (
        float(np.median(col_gap_values))
        if col_gap_values.size
        else distance_scale
    )

    row_gap_scale = max(row_gap_scale, eps)
    col_gap_scale = max(col_gap_scale, eps)

    row_regret = np.exp(
        -np.minimum(row_excess / row_gap_scale, 60.0)
    )
    col_regret = np.exp(
        -np.minimum(col_excess / col_gap_scale, 60.0)
    )

    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    temperature = max(0.22 * distance_scale, eps)
    endpoint_excess = 0.5 * (row_excess + col_excess)

    logits = np.zeros_like(d)
    np.divide(
        endpoint_excess,
        temperature,
        out=logits,
        where=valid,
    )
    np.minimum(logits, 60.0, out=logits)

    affinity = np.exp(-logits)
    affinity[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for iteration in range(64):
            row_mass = affinity @ col_scale
            np.divide(
                1.0,
                row_mass,
                out=row_scale,
                where=row_mass > 0.0,
            )
            row_scale[row_mass <= 0.0] = 0.0

            col_mass = affinity.T @ row_scale
            np.divide(
                1.0,
                col_mass,
                out=col_scale,
                where=col_mass > 0.0,
            )
            col_scale[col_mass <= 0.0] = 0.0

            if iteration >= 7:
                balanced = (
                    affinity
                    * row_scale[:, None]
                    * col_scale[None, :]
                )
                row_error = np.max(
                    np.abs(
                        np.sum(balanced, axis=1)[row_count > 0] - 1.0
                    ),
                    initial=0.0,
                )
                col_error = np.max(
                    np.abs(
                        np.sum(balanced, axis=0)[col_count > 0] - 1.0
                    ),
                    initial=0.0,
                )
                if max(row_error, col_error) <= 1e-4:
                    break

    balanced = affinity * row_scale[:, None] * col_scale[None, :]

    result = balanced * np.power(regret + eps, 0.85)
    result[~valid] = 0.0
    result[~np.isfinite(result)] = 0.0

    maximum = float(np.max(result))
    if maximum > 0.0 and np.isfinite(maximum):
        result /= maximum

    np.fill_diagonal(result, 0.0)
    return result.astype(float, copy=False)
