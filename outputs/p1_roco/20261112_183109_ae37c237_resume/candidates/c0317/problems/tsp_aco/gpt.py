import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return entropy-controlled edge-priority indicators for stochastic TSP sampling."""
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

    positive_values = d[valid]
    positive_values = positive_values[positive_values > 0.0]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(d[valid]))

    global_scale = max(global_scale, np.finfo(float).eps)

    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    k = min(4, n - 1)
    nearest = sorted_rows[:, :k]
    nearest_finite = np.isfinite(nearest)
    nearest_sum = np.where(nearest_finite, nearest, 0.0).sum(axis=1)
    nearest_count = nearest_finite.sum(axis=1)

    local_scale = np.divide(
        nearest_sum,
        np.maximum(nearest_count, 1),
        out=np.full(n, global_scale, dtype=float),
        where=nearest_count > 0,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(np.isfinite(second), second, 2.0 * safe_best)

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )

    endpoint_urgency = 1.0 + 0.12 * regret

    endpoint_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    scale = 0.70 * endpoint_scale + 0.30 * global_scale
    scale = np.maximum(scale, np.finfo(float).eps)

    normalized_distance = np.divide(
        d,
        scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_ratio = np.divide(
        d,
        local_scale[:, None],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        local_scale[None, :],
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    log_distance_quality = -np.clip(0.95 * normalized_distance, 0.0, 50.0)
    log_mutual_quality = -0.24 * np.clip(
        row_ratio + col_ratio,
        0.0,
        50.0,
    )

    row_ranks = np.zeros_like(d, dtype=float)
    col_ranks = np.zeros_like(d, dtype=float)

    row_ranks[valid] = (
        np.argsort(
            np.argsort(np.where(valid, d, np.inf), axis=1),
            axis=1,
        )[valid]
        + 1.0
    )
    col_ranks[valid] = (
        np.argsort(
            np.argsort(np.where(valid, d, np.inf), axis=0),
            axis=0,
        )[valid]
        + 1.0
    )

    log_rank_quality = -0.07 * (
        np.log1p(row_ranks) + np.log1p(col_ranks)
    )

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

    degree_feasibility = 1.0 / np.sqrt(
        np.maximum(row_competition[:, None], 1.0)
        * np.maximum(col_competition[None, :], 1.0)
    )
    log_feasibility = 0.08 * np.log1p(degree_feasibility)

    log_score = (
        log_distance_quality
        + log_mutual_quality
        + log_rank_quality
        + log_feasibility
        + np.log(endpoint_urgency[:, None])
        + np.log(endpoint_urgency[None, :])
    )
    log_score[~valid] = -np.inf

    row_max = np.max(log_score, axis=1, keepdims=True)
    stabilized = np.exp(
        np.clip(log_score - row_max, -60.0, 0.0)
    )
    stabilized[~valid] = 0.0

    counts = valid.sum(axis=1, keepdims=True).astype(float)
    totals = stabilized.sum(axis=1, keepdims=True)

    core = np.divide(
        stabilized,
        totals,
        out=np.zeros_like(stabilized),
        where=totals > 0.0,
    )

    safe_core = np.where(valid, core, 0.0)
    entropy = -np.sum(
        np.where(safe_core > 0.0, safe_core * np.log(safe_core), 0.0),
        axis=1,
        keepdims=True,
    )

    max_entropy = np.log(np.maximum(counts, 1.0))
    concentration = np.divide(
        max_entropy - entropy,
        np.maximum(max_entropy, np.finfo(float).eps),
        out=np.zeros_like(entropy),
        where=max_entropy > 0.0,
    )
    concentration = np.clip(concentration, 0.0, 1.0)

    mix = 0.025 + 0.10 * concentration
    uniform = np.divide(
        valid.astype(float),
        np.maximum(counts, 1.0),
    )

    result = (1.0 - mix) * core + mix * uniform
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    totals = result.sum(axis=1, keepdims=True)
    return np.divide(
        result,
        totals,
        out=np.zeros_like(result),
        where=totals > 0.0,
    )
