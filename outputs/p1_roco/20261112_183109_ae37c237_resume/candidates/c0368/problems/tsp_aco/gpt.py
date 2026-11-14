import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return balanced local and long-term heuristic indicators for TSP."""
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

    sorted_rows = np.sort(work, axis=1)
    k = min(4, n - 1)
    nearest = sorted_rows[:, :k]
    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)

    local_sum = np.where(finite_nearest, nearest, 0.0).sum(axis=1)
    local_scale = np.divide(
        local_sum,
        np.maximum(counts, 1),
        out=np.ones(n, dtype=float),
        where=counts > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    best = sorted_rows[:, 0]
    second = sorted_rows[:, 1] if n > 2 else best.copy()

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        2.0 * safe_best,
    )

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )
    urgency = 1.0 + 0.05 * regret

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    row_ratio = np.divide(
        d,
        row_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else float(np.median(finite_values))
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        distance_quality = np.exp(
            -np.clip(normalized_distance, 0.0, 50.0) / 0.82
        )

        row_quality = np.exp(
            -np.clip(row_ratio, 0.0, 50.0) / 0.88
        )
        col_quality = np.exp(
            -np.clip(col_ratio, 0.0, 50.0) / 0.88
        )

        geometric_quality = np.sqrt(
            np.maximum(row_quality, 0.0)
            * np.maximum(col_quality, 0.0)
        )
        arithmetic_quality = 0.5 * (row_quality + col_quality)

        directional_quality = (
            0.78 * geometric_quality
            + 0.22 * arithmetic_quality
        )

        base_score = (
            distance_quality
            * directional_quality
            * urgency[:, None]
        )

        global_quality = np.exp(
            -np.clip(d / global_scale, 0.0, 50.0) / 1.8
        )
        global_quality = np.where(valid, global_quality, 0.0)

        row_near_count = np.sum(
            valid & (d <= safe_second[:, None]),
            axis=1,
            dtype=float,
        )
        col_near_count = np.sum(
            valid & (d <= safe_second[None, :]),
            axis=0,
            dtype=float,
        )

        row_scarcity = 1.0 / np.sqrt(
            np.maximum(row_near_count, 1.0)
        )
        col_scarcity = 1.0 / np.sqrt(
            np.maximum(col_near_count, 1.0)
        )
        scarcity = (
            row_scarcity[:, None] + col_scarcity[None, :]
        )
        scarcity /= max(float(np.max(scarcity)), 1.0)

        reverse_quality = np.exp(
            -np.clip(d.T / global_scale, 0.0, 50.0) / 1.55
        )
        reverse_quality = np.where(valid.T, reverse_quality, 0.0)

        mutual_quality = np.sqrt(
            np.maximum(row_quality, 0.0)
            * np.maximum(reverse_quality, 0.0)
        )
        mutual_quality = np.where(valid, mutual_quality, 0.0)

        endpoint_regret = 0.5 * (
            (safe_second - safe_best)[:, None]
            + (safe_second - safe_best)[None, :]
        )
        regret_scale = np.maximum(
            global_scale
            + 0.5 * (safe_second[:, None] + safe_second[None, :]),
            eps,
        )
        opportunity_factor = np.exp(
            -np.clip(endpoint_regret / regret_scale, 0.0, 8.0)
            / 2.8
        )

        long_term_factor = (
            1.0
            + 0.06 * global_quality
            + 0.04 * mutual_quality
            + 0.035 * scarcity
        )
        long_term_factor *= (
            0.985 + 0.015 * opportunity_factor
        )

        score = base_score * long_term_factor

        positive_scores = np.where(valid, score, 0.0)
        row_max = positive_scores.max(axis=1, keepdims=True)
        exploration_floor = 0.015 * row_max
        score = np.where(
            valid,
            score + exploration_floor,
            0.0,
        )

    score = np.maximum(score, 0.0)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
