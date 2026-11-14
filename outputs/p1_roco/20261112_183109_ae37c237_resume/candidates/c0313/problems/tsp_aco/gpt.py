import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized stochastic priorities for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else float(np.median(d[valid]))
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    row_scale = np.empty(n, dtype=float)
    col_scale = np.empty(n, dtype=float)

    for i in range(n):
        values = d[i, valid[i] & (d[i] > 0.0)]
        row_scale[i] = (
            float(np.median(values))
            if values.size
            else global_scale
        )

    for j in range(n):
        values = d[valid[:, j] & (d[:, j] > 0.0), j]
        col_scale[j] = (
            float(np.median(values))
            if values.size
            else global_scale
        )

    row_scale = np.maximum(row_scale, eps)
    col_scale = np.maximum(col_scale, eps)

    row_ratio = np.divide(
        d,
        row_scale[:, None],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale[None, :],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    global_ratio = np.divide(
        d,
        global_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_affinity = np.exp(-np.clip(row_ratio, 0.0, 50.0))
    col_affinity = np.exp(-np.clip(col_ratio, 0.0, 50.0))
    global_affinity = np.exp(-np.clip(global_ratio, 0.0, 50.0) / 1.6)

    row_affinity[~valid] = 0.0
    col_affinity[~valid] = 0.0
    global_affinity[~valid] = 0.0

    row_sum = row_affinity.sum(axis=1, keepdims=True)
    col_sum = col_affinity.sum(axis=0, keepdims=True)

    row_prob = np.divide(
        row_affinity,
        np.maximum(row_sum, eps),
        out=np.zeros_like(d),
        where=row_sum > 0.0,
    )
    col_prob = np.divide(
        col_affinity,
        np.maximum(col_sum, eps),
        out=np.zeros_like(d),
        where=col_sum > 0.0,
    )

    row_count = valid.sum(axis=1).astype(float)
    col_count = valid.sum(axis=0).astype(float)

    row_entropy = -np.sum(
        np.where(
            row_prob > 0.0,
            row_prob * np.log(np.maximum(row_prob, eps)),
            0.0,
        ),
        axis=1,
    )
    col_entropy = -np.sum(
        np.where(
            col_prob > 0.0,
            col_prob * np.log(np.maximum(col_prob, eps)),
            0.0,
        ),
        axis=0,
    )

    row_entropy = np.divide(
        row_entropy,
        np.log(np.maximum(row_count, 2.0)),
        out=np.zeros(n, dtype=float),
        where=row_count > 1.0,
    )
    col_entropy = np.divide(
        col_entropy,
        np.log(np.maximum(col_count, 2.0)),
        out=np.zeros(n, dtype=float),
        where=col_count > 1.0,
    )
    row_entropy = np.clip(row_entropy, 0.0, 1.0)
    col_entropy = np.clip(col_entropy, 0.0, 1.0)

    row_mix = 0.08 + 0.14 * row_entropy
    col_mix = 0.08 + 0.14 * col_entropy

    row_quality = (
        (1.0 - row_mix[:, None]) * row_affinity
        + row_mix[:, None] * global_affinity
    )
    col_quality = (
        (1.0 - col_mix[None, :]) * col_affinity
        + col_mix[None, :] * global_affinity
    )
    quality = np.sqrt(row_quality * col_quality)
    quality[~valid] = 0.0

    row_residual = np.maximum(row_quality.sum(axis=1, keepdims=True) - row_quality, 0.0)
    col_residual = np.maximum(col_quality.sum(axis=0, keepdims=True) - col_quality, 0.0)

    row_reserve = np.divide(
        row_residual,
        np.maximum(row_quality.sum(axis=1, keepdims=True), eps),
        out=np.zeros_like(d),
        where=row_quality.sum(axis=1, keepdims=True) > 0.0,
    )
    col_reserve = np.divide(
        col_residual,
        np.maximum(col_quality.sum(axis=0, keepdims=True), eps),
        out=np.zeros_like(d),
        where=col_quality.sum(axis=0, keepdims=True) > 0.0,
    )

    future_factor = 1.0 + 0.10 * (
        1.0 - np.sqrt(np.clip(row_reserve * col_reserve, 0.0, 1.0))
    )

    k = min(5, n - 1)
    neighborhood = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[np.argsort(d[i, candidates], kind="stable")]
            neighborhood[i, order[:k]] = 1.0

    overlap = neighborhood @ neighborhood.T
    counts = neighborhood.sum(axis=1)
    denominator = np.sqrt(
        np.maximum(counts[:, None], 1.0)
        * np.maximum(counts[None, :], 1.0)
    )
    overlap = np.divide(
        overlap,
        denominator,
        out=np.zeros_like(overlap),
        where=denominator > 0.0,
    )
    overlap = 0.5 * (overlap + overlap.T)
    np.fill_diagonal(overlap, 1.0)

    bridge_factor = 1.0 + 0.08 * (1.0 - overlap)
    cluster_factor = np.exp(-0.035 * overlap)

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.empty((n, n), dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = (
        np.arange(n, dtype=float)[None, :] + 1.0
    )

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.empty((n, n), dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = (
        np.arange(n, dtype=float)[:, None] + 1.0
    )

    rank_factor = 1.0 / np.sqrt(
        (1.0 + 0.045 * np.maximum(row_rank - 1.0, 0.0))
        * (1.0 + 0.045 * np.maximum(col_rank - 1.0, 0.0))
    )

    core = (
        quality
        * rank_factor
        * bridge_factor
        * cluster_factor
        * future_factor
    )
    core[~valid] = 0.0

    row_floor = np.clip(0.025 + 0.065 * row_entropy, 0.025, 0.09)
    col_floor = np.clip(0.025 + 0.065 * col_entropy, 0.025, 0.09)

    uniform_row = np.divide(
        valid.astype(float),
        np.maximum(row_count[:, None], 1.0),
    )
    uniform_col = np.divide(
        valid.astype(float),
        np.maximum(col_count[None, :], 1.0),
    )

    exploratory = (
        0.5
        * (
            (1.0 - row_floor[:, None]) * row_prob
            + row_floor[:, None] * uniform_row
        )
        + 0.5
        * (
            (1.0 - col_floor[None, :]) * col_prob
            + col_floor[None, :] * uniform_col
        )
    )

    concentration = np.maximum(
        1.0 - row_entropy[:, None],
        1.0 - col_entropy[None, :],
    )
    global_mix = np.clip(0.06 + 0.10 * concentration, 0.06, 0.16)

    score = (1.0 - global_mix) * core + global_mix * exploratory
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
