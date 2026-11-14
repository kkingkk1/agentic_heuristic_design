import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return adaptive, row-normalized edge-priority indicators for stochastic TSP sampling."""
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

    positive_values = d[valid & (d > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    k = min(5, n - 1)
    nearest = np.sort(work, axis=1)[:, :k]
    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)

    nearest_sum = np.where(finite_nearest, nearest, 0.0).sum(axis=1)
    local_scale = np.divide(
        nearest_sum,
        np.maximum(counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    adaptive_scale = np.sqrt(local_scale * global_scale)
    normalized = np.divide(
        d,
        np.sqrt(adaptive_scale[:, None] * adaptive_scale[None, :]),
        out=np.full_like(d, np.inf),
        where=valid,
    )

    row_dispersion = np.divide(
        np.where(finite_nearest, nearest, 0.0).std(axis=1),
        np.maximum(local_scale, eps),
    )
    decay = 0.92 + 0.18 * np.clip(row_dispersion, 0.0, 1.0)
    distance_quality = np.exp(
        -np.clip(decay[:, None] * normalized, 0.0, 50.0)
    )

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

    row_support = np.exp(-np.clip(row_ratio, 0.0, 50.0))
    col_support = np.exp(-np.clip(col_ratio, 0.0, 50.0))
    mutual_support = np.sqrt(row_support * col_support)

    row_order = np.argsort(work, axis=1)
    col_order = np.argsort(work, axis=0)

    row_rank = np.empty_like(row_order, dtype=float)
    col_rank = np.empty_like(col_order, dtype=float)

    row_rank[row_order, np.arange(n)[:, None]] = (
        np.arange(n, dtype=float)[None, :] + 1.0
    )
    col_rank[col_order, np.arange(n)[None, :]] = (
        np.arange(n, dtype=float)[:, None] + 1.0
    )

    rank_support = 1.0 / np.sqrt(
        (1.0 + 0.10 * row_rank) * (1.0 + 0.10 * col_rank)
    )

    second = np.sort(work, axis=1)[:, 1] if n > 2 else np.sort(work, axis=1)[:, 0]
    best = np.sort(work, axis=1)[:, 0]

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        safe_best + local_scale,
    )

    endpoint_regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    endpoint_bonus = 1.0 + 0.08 * endpoint_regret[:, None]
    endpoint_bonus *= 1.0 + 0.08 * endpoint_regret[None, :]

    row_competition = np.sum(
        valid & (d <= safe_second[:, None]),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (d <= safe_second[None, :]),
        axis=0,
        dtype=float,
    )

    scarcity = 1.0 + 0.035 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )

    score = (
        distance_quality
        * mutual_support
        * rank_support
        * endpoint_bonus
        * scarcity
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        score,
        totals,
        out=np.zeros_like(score),
        where=totals > 0.0,
    )

    valid_counts = valid.sum(axis=1, keepdims=True).astype(float)
    uniform = np.divide(
        valid.astype(float),
        np.maximum(valid_counts, 1.0),
    )

    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0),
        axis=1,
        keepdims=True,
    )
    max_entropy = np.log(np.maximum(valid_counts, 2.0))
    concentration = 1.0 - np.divide(
        entropy,
        np.maximum(max_entropy, eps),
    )

    exploration = 0.015 + 0.045 * np.clip(concentration, 0.0, 1.0)
    probabilities = (
        (1.0 - exploration) * probabilities
        + exploration * uniform
    )

    probabilities[~valid] = 0.0
    np.fill_diagonal(probabilities, 0.0)

    final_totals = probabilities.sum(axis=1, keepdims=True)
    return np.divide(
        probabilities,
        final_totals,
        out=np.zeros_like(probabilities),
        where=final_totals > 0.0,
    )
