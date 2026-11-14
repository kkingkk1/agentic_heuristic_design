import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
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

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    ordered = np.sort(work, axis=1)
    best = ordered[:, 0]

    if n > 2:
        second = ordered[:, 1]
        third = ordered[:, 2]
    else:
        second = best.copy()
        third = best.copy()

    safe_best = np.where(np.isfinite(best), best, 1.0)
    safe_second = np.where(
        np.isfinite(second), second, 2.0 * np.maximum(safe_best, eps)
    )
    safe_third = np.where(
        np.isfinite(third), third, 2.0 * np.maximum(safe_second, eps)
    )

    local_scale = np.maximum(
        (safe_best + safe_second + safe_third) / 3.0,
        eps,
    )

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.8 * regret

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

    row_quality = np.exp(-np.clip(row_ratio, 0.0, 60.0))
    col_quality = np.exp(-np.clip(col_ratio, 0.0, 60.0))

    geometric_quality = np.exp(
        -np.clip(
            d / np.sqrt(local_scale[:, None] * local_scale[None, :]),
            0.0,
            60.0,
        )
    )

    row_order = np.argsort(np.argsort(work, axis=1), axis=1)
    col_order = np.argsort(np.argsort(work, axis=0), axis=0)

    row_rank = row_order.astype(float)
    col_rank = col_order.astype(float)

    rank_quality = np.exp(
        -0.35 * (row_rank + col_rank)
        / max(float(n - 1), 1.0)
    )

    row_best = row_rank == 0.0
    col_best = col_rank == 0.0
    mutual_bonus = 1.0 + 0.22 * (row_best & col_best)
    one_sided_bonus = 1.0 + 0.06 * (row_best ^ col_best)

    local_quality = np.sqrt(row_quality * col_quality)

    row_strength = np.sum(
        np.where(valid, np.exp(-np.clip(row_ratio, 0.0, 60.0)), 0.0),
        axis=1,
    )
    row_count = np.maximum(valid.sum(axis=1), 1)
    concentration = np.divide(
        np.max(
            np.where(valid, np.exp(-np.clip(row_ratio, 0.0, 60.0)), 0.0),
            axis=1,
        ),
        np.maximum(row_strength, eps),
    )
    concentration = np.clip(concentration, 0.0, 1.0)

    exploration_floor = (
        0.025
        + 0.09 * concentration
    )[:, None]
    exploration_floor = exploration_floor / row_count[:, None]

    relative_excess = np.maximum(
        np.maximum(row_ratio, col_ratio) - 1.0,
        0.0,
    )
    bridge_signal = np.clip(relative_excess / 3.0, 0.0, 1.0)
    bridge_penalty = 1.0 - 0.12 * bridge_signal

    score = (
        geometric_quality
        * local_quality
        * rank_quality
        * mutual_bonus
        * one_sided_bonus
        * urgency[:, None]
        * urgency[None, :]
        * bridge_penalty
    )

    score = np.where(valid, score, 0.0)
    score += np.where(valid, exploration_floor, 0.0)
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )
