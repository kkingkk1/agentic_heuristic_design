import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based, scarcity-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    result = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    safe = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1).astype(float)
    col_count = valid.sum(axis=0).astype(float)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.full((n, n), n, dtype=float)
    col_rank = np.full((n, n), n, dtype=float)

    row_positions = np.broadcast_to(
        np.arange(n, dtype=float),
        (n, n),
    )
    col_positions = np.broadcast_to(
        np.arange(n, dtype=float)[:, None],
        (n, n),
    )

    np.put_along_axis(row_rank, row_order, row_positions, axis=1)
    np.put_along_axis(col_rank, col_order, col_positions, axis=0)

    row_rank = np.where(valid, row_rank, 0.0)
    col_rank = np.where(valid, col_rank, 0.0)

    row_priority = np.power(1.0 + row_rank, -1.35)
    col_priority = np.power(1.0 + col_rank, -1.15)

    source_scarcity = np.divide(
        1.0,
        np.sqrt(np.maximum(row_count, 1.0)),
    )
    target_scarcity = np.divide(
        1.0,
        np.sqrt(np.maximum(col_count, 1.0)),
    )

    scarcity = (
        source_scarcity[:, None] * target_scarcity[None, :]
    )

    valid_values = d[valid]
    positive = valid_values[valid_values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    baseline = row_best[:, None] + col_best[None, :] + scale
    relative_detour = np.divide(
        d,
        baseline,
        out=np.zeros_like(d),
        where=valid,
    )

    detour_penalty = np.reciprocal(
        1.0 + np.maximum(relative_detour - 1.0, 0.0)
    )

    score = (
        row_priority
        * col_priority
        * (0.65 + 0.35 * scarcity)
        * detour_penalty
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
