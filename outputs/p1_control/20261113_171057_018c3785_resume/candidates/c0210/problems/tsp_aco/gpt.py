import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank- and neighborhood-based priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    eps = np.finfo(float).eps
    safe = np.where(valid, d, np.inf)

    order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.empty_like(order, dtype=float)
    row_rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    row_rank = np.where(valid, row_rank, 0.0)

    reverse_order = np.argsort(safe, axis=0, kind="stable")
    col_rank = np.empty_like(reverse_order, dtype=float)
    col_rank[reverse_order, np.arange(n)[None, :]] = np.arange(n)[:, None]
    col_rank = np.where(valid, col_rank, 0.0)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    sorted_rows = np.sort(safe, axis=1)
    sorted_cols = np.sort(safe, axis=0)

    row_index = np.maximum((row_count - 1) // 2, 0)
    col_index = np.maximum((col_count - 1) // 2, 0)

    row_median = sorted_rows[np.arange(n), row_index]
    col_median = sorted_cols[col_index, np.arange(n)]

    finite_values = d[valid]
    global_scale = float(np.median(finite_values))
    global_scale = max(global_scale, eps)

    row_median = np.where(
        np.isfinite(row_median) & (row_median > 0.0),
        row_median,
        global_scale,
    )
    col_median = np.where(
        np.isfinite(col_median) & (col_median > 0.0),
        col_median,
        global_scale,
    )

    local_scale = np.sqrt(
        np.maximum(row_median[:, None] * col_median[None, :], eps)
    )

    distance_ratio = np.divide(
        d,
        local_scale,
        out=np.zeros_like(d),
        where=valid,
    )

    distance_component = 1.0 / (1.0 + distance_ratio)

    outgoing_rank = 1.0 / np.power(1.0 + row_rank, 1.35)
    incoming_rank = 1.0 / np.power(1.0 + col_rank, 1.15)

    rank_component = np.sqrt(outgoing_rank * incoming_rank)

    candidate_width = max(2, int(np.ceil(np.sqrt(max(n - 1, 1)))))
    candidate_mask = (row_rank < candidate_width) | (
        col_rank < candidate_width
    )

    mutual_nearest = (
        (row_rank == 0.0)
        & (col_rank == 0.0)
        & valid
    ).astype(float)

    mutual_candidate = (
        (row_rank < candidate_width)
        & (col_rank < candidate_width)
        & valid
    ).astype(float)

    scarcity = 1.0 / (
        1.0
        + np.divide(
            np.maximum(d - row_median[:, None], 0.0),
            row_median[:, None],
            out=np.zeros_like(d),
            where=valid,
        )
    )

    score = (
        0.46 * rank_component
        + 0.28 * distance_component
        + 0.14 * scarcity
        + 0.08 * mutual_candidate
        + 0.18 * mutual_nearest
    )

    score = np.where(candidate_mask, score, 0.42 * score)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
