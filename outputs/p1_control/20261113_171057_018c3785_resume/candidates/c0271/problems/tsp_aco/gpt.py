import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based directed TSP edge priors."""
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

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, d, np.inf)
    row_count = valid.sum(axis=1).astype(float)
    col_count = valid.sum(axis=0).astype(float)

    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    positions = np.arange(n, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = positions[None, :]
    col_rank[col_order, np.arange(n)[None, :]] = positions[:, None]

    row_den = np.maximum(row_count[:, None] - 1.0, 1.0)
    col_den = np.maximum(col_count[None, :] - 1.0, 1.0)

    row_quality = 1.0 / (1.0 + row_rank / row_den)
    col_quality = 1.0 / (1.0 + col_rank / col_den)

    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    reverse_valid = valid.T
    reverse_rank = np.zeros((n, n), dtype=float)
    reverse_rank[valid] = col_rank.T[valid]
    reverse_den = np.maximum(col_count[:, None] - 1.0, 1.0)
    reverse_quality = 1.0 / (1.0 + reverse_rank / reverse_den)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    relative_cost = np.zeros_like(d, dtype=float)
    np.divide(
        np.log1p(np.maximum(d, 0.0) / scale),
        np.log1p(np.maximum(safe, 0.0) / scale),
        out=relative_cost,
        where=valid & np.isfinite(safe),
    )
    magnitude_quality = np.exp(-np.clip(relative_cost, 0.0, 20.0))

    nearest_row = valid & (row_rank == 0.0)
    nearest_col = valid & (col_rank == 0.0)
    mutual_bonus = np.where(nearest_row & nearest_col, 1.8, 1.0)

    destination_access = np.sqrt(
        np.divide(
            col_count[None, :],
            max(float(n - 1), 1.0),
        )
    )
    destination_access = np.clip(destination_access, 0.25, 1.0)

    score = (
        row_quality
        * col_quality
        * reverse_quality
        * magnitude_quality
        * mutual_bonus
        * destination_access
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
