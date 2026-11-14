import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a feasibility-preserving, bidirectionally rank-aware TSP prior."""
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

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)

    row_min = np.min(safe, axis=1)
    col_min = np.min(safe, axis=0)

    row_min = np.where(np.isfinite(row_min), row_min, 0.0)
    col_min = np.where(np.isfinite(col_min), col_min, 0.0)

    row_counts = valid.sum(axis=1)
    col_counts = valid.sum(axis=0)

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

    finite_row_gaps = row_gap[np.isfinite(row_gap) & (row_gap > 0.0)]
    finite_col_gaps = col_gap[np.isfinite(col_gap) & (col_gap > 0.0)]

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
        np.isfinite(row_gap) & (row_gap > 0.0),
        row_gap,
        row_gap_fallback,
    )
    col_gap = np.where(
        np.isfinite(col_gap) & (col_gap > 0.0),
        col_gap,
        col_gap_fallback,
    )

    row_delta = np.where(
        valid,
        np.maximum(d - row_min[:, None], 0.0),
        0.0,
    )
    col_delta = np.where(
        valid,
        np.maximum(d - col_min[None, :], 0.0),
        0.0,
    )

    row_regret = np.exp(
        -np.clip(row_delta / max(row_gap_fallback, np.finfo(float).eps), 0.0, 50.0)
    )
    col_regret = np.exp(
        -np.clip(col_delta / max(col_gap_fallback, np.finfo(float).eps), 0.0, 50.0)
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

            average_rank = 0.5 * (start + end - 1)
            output[ordered[start:end]] = average_rank
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
        np.maximum(row_gap, np.finfo(float).eps)[:, None]
        * np.maximum(col_gap, np.finfo(float).eps)[None, :]
    )
    local_temperature = np.maximum(local_temperature, 0.18 * global_scale)

    distance_affinity = np.exp(
        -np.clip(
            np.divide(
                d,
                local_temperature,
                out=np.zeros_like(d),
                where=np.isfinite(local_temperature) & (local_temperature > 0.0),
            ),
            0.0,
            60.0,
        )
    )
    distance_affinity[~valid] = 0.0

    base = distance_affinity * (
        0.58 * mutual_rank + 0.42 * regret
    )
    base[~valid] = 0.0

    balanced = base.copy()

    for _ in range(64):
        previous = balanced.copy()

        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced = np.divide(
            balanced,
            row_sum,
            out=np.zeros_like(balanced),
            where=row_sum > 0.0,
        )

        col_sum = balanced.sum(axis=0, keepdims=True)
        balanced = np.divide(
            balanced,
            col_sum,
            out=np.zeros_like(balanced),
            where=col_sum > 0.0,
        )

        if np.max(np.abs(balanced - previous)) < 1e-10:
            break

    refinement = (
        0.62 * distance_affinity
        + 0.38 * np.sqrt(np.maximum(mutual_rank * regret, 0.0))
    )

    score = balanced * np.power(np.maximum(refinement, 0.0), 0.85)
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
