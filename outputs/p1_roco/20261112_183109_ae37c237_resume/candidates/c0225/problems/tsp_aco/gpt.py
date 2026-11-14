import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative row-normalized edge-priority indicators for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)
    ordered = np.sort(work, axis=1)

    best = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else best
    third = ordered[:, 2] if n > 2 else second

    finite_best = np.isfinite(best)
    safe_best = np.where(finite_best, best, 1.0)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)
    safe_third = np.where(np.isfinite(third), third, 2.0 * safe_second)

    local_scale = np.maximum(
        (safe_best + safe_second + safe_third) / 3.0,
        np.finfo(float).eps,
    )

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.18 * regret

    row_ratio = np.divide(
        d,
        local_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        local_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    row_quality = np.exp(-np.minimum(row_ratio, 60.0))
    col_quality = np.exp(-np.minimum(col_ratio, 60.0))
    mutual_quality = np.sqrt(row_quality * col_quality)

    geometric_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    geometric_ratio = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )
    distance_quality = np.exp(-np.minimum(geometric_ratio, 60.0))

    nearest = valid & (d <= best[:, None])
    nearest_bonus = 1.0 + 0.08 * nearest.astype(float)

    result = (
        distance_quality
        * mutual_quality
        * nearest_bonus
        * urgency[:, None]
        * urgency[None, :]
    )

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    if np.allclose(d, d.T, equal_nan=True):
        result = 0.5 * (result + result.T)

    totals = result.sum(axis=1, keepdims=True)
    return np.divide(
        result,
        totals,
        out=np.zeros_like(result),
        where=totals > 0.0,
    )
