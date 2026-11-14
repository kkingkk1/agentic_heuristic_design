import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a balanced, bidirectionally rank-aware TSP edge prior."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    safe = np.where(valid, d, np.inf)

    row_count = valid.sum(axis=1)
    col_count = valid.sum(axis=0)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)
    row_min = np.where(row_count > 0, row_min, 0.0)
    col_min = np.where(col_count > 0, col_min, 0.0)

    row_second = np.full(n, np.inf, dtype=float)
    col_second = np.full(n, np.inf, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        if values.size >= 2:
            row_second[i] = np.partition(values, 1)[1]

    for j in range(n):
        values = d[valid[:, j], j]
        if values.size >= 2:
            col_second[j] = np.partition(values, 1)[1]

    row_gap = row_second - row_min
    col_gap = col_second - col_min

    finite_row_gaps = row_gap[np.isfinite(row_gap) & (row_gap > eps)]
    finite_col_gaps = col_gap[np.isfinite(col_gap) & (col_gap > eps)]

    row_gap_fallback = (
        float(np.median(finite_row_gaps))
        if finite_row_gaps.size
        else global_scale
    )
    col_gap_fallback = (
        float(np.median(finite_col_gaps))
        if finite_col_gaps.size
        else global_scale
    )

    row_gap = np.where(
        np.isfinite(row_gap) & (row_gap > eps),
        row_gap,
        row_gap_fallback,
    )
    col_gap = np.where(
        np.isfinite(col_gap) & (col_gap > eps),
        col_gap,
        col_gap_fallback,
    )

    row_delta = np.maximum(d - row_min[:, None], 0.0)
    col_delta = np.maximum(d - col_min[None, :], 0.0)
    row_delta[~valid] = 0.0
    col_delta[~valid] = 0.0

    row_regret = np.exp(
        -np.clip(
            row_delta / max(row_gap_fallback, eps),
            0.0,
            50.0,
        )
    )
    col_regret = np.exp(
        -np.clip(
            col_delta / max(col_gap_fallback, eps),
            0.0,
            50.0,
        )
    )
    regret = np.sqrt(row_regret * col_regret)
    regret[~valid] = 0.0

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    def assign_average_ranks(values, mask, output):
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return

        ordered = indices[np.argsort(values[indices], kind="stable")]
        ordered_values = values[ordered]

        start = 0
        total = ordered.size
        while start < total:
            end = start + 1
            while end < total and ordered_values[end] == ordered_values[start]:
                end += 1

            output[ordered[start:end]] = 0.5 * (start + end - 1)
            start = end

    for i in range(n):
        assign_average_ranks(d[i], valid[i], row_rank[i])

    for j in range(n):
        assign_average_ranks(d[:, j], valid[:, j], col_rank[:, j])

    row_rank_score = np.divide(
        1.0,
        1.0 + row_rank,
        out=np.zeros_like(row_rank),
        where=valid,
    )
    col_rank_score = np.divide(
        1.0,
        1.0 + col_rank,
        out=np.zeros_like(col_rank),
        where=valid,
    )

    mutual_rank = np.sqrt(row_rank_score * col_rank_score)
    mutual_rank[~valid] = 0.0

    local_temperature = np.sqrt(
        np.maximum(row_gap, eps)[:, None]
        * np.maximum(col_gap, eps)[None, :]
    )
    local_temperature = np.maximum(
        local_temperature,
        0.18 * global_scale,
    )

    distance_affinity = np.exp(
        -np.clip(
            np.divide(
                d,
                local_temperature,
                out=np.zeros_like(d),
                where=local_temperature > 0.0,
            ),
            0.0,
            60.0,
        )
    )
    distance_affinity[~valid] = 0.0

    affinity = distance_affinity * (
        0.50 * mutual_rank
        + 0.30 * regret
        + 0.20 * np.sqrt(np.maximum(mutual_rank * regret, 0.0))
    )
    affinity[~valid] = 0.0

    row_scale = np.ones(n, dtype=float)
    col_scale = np.ones(n, dtype=float)

    row_sum = affinity.sum(axis=1)
    row_scale = np.divide(
        1.0,
        row_sum,
        out=np.zeros_like(row_sum),
        where=row_sum > 0.0,
    )

    col_sum = affinity.T @ row_scale
    col_scale = np.divide(
        1.0,
        col_sum,
        out=np.zeros_like(col_sum),
        where=col_sum > 0.0,
    )

    for _ in range(12):
        row_sum = affinity @ col_scale
        row_scale = np.divide(
            1.0,
            row_sum,
            out=np.zeros_like(row_sum),
            where=row_sum > 0.0,
        )

        col_sum = affinity.T @ row_scale
        col_scale = np.divide(
            1.0,
            col_sum,
            out=np.zeros_like(col_sum),
            where=col_sum > 0.0,
        )

    score = affinity * row_scale[:, None] * col_scale[None, :]
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
