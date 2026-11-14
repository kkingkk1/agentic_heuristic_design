import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative row-normalized edge-priority indicators for TSP."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return score

    work = np.where(valid, d, np.inf)
    ordered = np.sort(work, axis=1)

    best = ordered[:, 0]
    second = ordered[:, 1] if n > 2 else best.copy()
    third = ordered[:, 2] if n > 2 else second.copy()

    finite_best = np.isfinite(best)
    safe_best = np.where(finite_best, best, 1.0)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)
    safe_third = np.where(np.isfinite(third), third, 2.0 * safe_second)

    eps = np.finfo(float).eps
    local_scale = np.maximum(
        (safe_best + safe_second + safe_third) / 3.0,
        eps,
    )

    regret = np.clip(
        (safe_second - safe_best) / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.10 * regret

    row_ratio = np.divide(
        d,
        local_scale[:, None],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        local_scale[None, :],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 60.0))
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 60.0))
    mutual_quality = np.sqrt(row_quality * col_quality)

    geometric_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    distance_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                geometric_scale,
                out=np.full_like(d, np.inf, dtype=float),
                where=valid,
            ),
            0.0,
            60.0,
        )
    )

    nearest_bonus = np.ones_like(d, dtype=float)
    nearest_bonus[valid & (d <= best[:, None])] = 1.04

    score = (
        distance_quality
        * mutual_quality
        * nearest_bonus
        * urgency[:, None]
        * urgency[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0,
    )
