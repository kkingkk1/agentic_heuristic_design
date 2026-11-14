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

    safe = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1).astype(float)
    col_count = valid.sum(axis=0).astype(float)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    positions = np.arange(n, dtype=float)

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    np.put_along_axis(
        row_rank,
        row_order,
        positions[None, :],
        axis=1,
    )
    np.put_along_axis(
        col_rank,
        col_order,
        positions[:, None],
        axis=0,
    )

    row_quality = np.zeros_like(d, dtype=float)
    col_quality = np.zeros_like(d, dtype=float)

    np.divide(
        row_count[:, None] - row_rank,
        row_count[:, None],
        out=row_quality,
        where=valid & (row_count[:, None] > 0.0),
    )
    np.divide(
        col_count[None, :] - col_rank,
        col_count[None, :],
        out=col_quality,
        where=valid & (col_count[None, :] > 0.0),
    )

    row_quality = np.maximum(row_quality, 0.0)
    col_quality = np.maximum(col_quality, 0.0)

    rank_agreement = 1.0 - np.abs(row_quality - col_quality)
    rank_agreement = np.clip(rank_agreement, 0.0, 1.0)

    scarcity = 1.0 / np.sqrt(
        np.maximum(row_count[:, None], 1.0)
        * np.maximum(col_count[None, :], 1.0)
    )
    scarcity /= np.max(scarcity[valid])

    score = (
        row_quality
        * col_quality
        * (0.5 + 0.5 * rank_agreement)
        * (0.75 + 0.25 * scarcity)
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
