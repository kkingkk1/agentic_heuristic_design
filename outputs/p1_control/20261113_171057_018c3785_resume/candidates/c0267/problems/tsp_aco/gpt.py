import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based directed TSP edge priors."""
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

    large = np.finfo(float).max
    masked = np.where(valid, d, large)

    row_order = np.argsort(masked, axis=1, kind="stable")
    col_order = np.argsort(masked, axis=0, kind="stable")

    row_rank = np.full((n, n), n, dtype=float)
    col_rank = np.full((n, n), n, dtype=float)

    positions = np.arange(n, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = positions[None, :]
    col_rank[col_order, np.arange(n)[None, :]] = positions[:, None]

    row_rank[~valid] = n
    col_rank[~valid] = n

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    candidate_width = max(1, int(np.ceil(np.sqrt(n))))
    row_width = np.minimum(candidate_width, np.maximum(row_count, 1))
    col_width = np.minimum(candidate_width, np.maximum(col_count, 1))

    row_candidate = valid & (
        row_rank < row_width[:, None]
    )
    col_candidate = valid & (
        col_rank < col_width[None, :]
    )

    row_weight = np.zeros_like(d, dtype=float)
    col_weight = np.zeros_like(d, dtype=float)

    np.divide(
        1.0,
        np.power(row_rank + 1.0, 0.85),
        out=row_weight,
        where=valid,
    )
    np.divide(
        1.0,
        np.power(col_rank + 1.0, 0.85),
        out=col_weight,
        where=valid,
    )

    score = row_weight * col_weight

    mutual = row_candidate & col_candidate
    score *= np.where(mutual, 1.35, 1.0)

    reciprocal = valid & np.isfinite(d.T)
    reverse_rank = np.zeros_like(d, dtype=float)
    reverse_rank[reciprocal] = 0.5 * (
        row_rank.T[reciprocal] + col_rank.T[reciprocal]
    )

    reciprocal_bonus = np.zeros_like(d, dtype=float)
    reciprocal_bonus[valid] = 1.0 / (1.0 + reverse_rank[valid])
    score *= 1.0 + 0.12 * reciprocal_bonus

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
