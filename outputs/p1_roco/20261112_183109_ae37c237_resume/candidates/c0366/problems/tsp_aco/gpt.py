import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return distance-aware, diverse, long-term edge priorities for TSP sampling."""
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

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        nonzero = finite_values[finite_values > 0.0]
        global_scale = float(np.mean(nonzero)) if nonzero.size else 1.0

    sorted_rows = np.sort(work, axis=1)
    best = sorted_rows[:, 0]

    if n > 2:
        second = sorted_rows[:, 1]
        third = sorted_rows[:, 2]
    else:
        second = best.copy()
        third = second.copy()

    safe_best = np.where(np.isfinite(best), best, global_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        np.maximum(safe_best, global_scale),
    )
    safe_third = np.where(
        np.isfinite(third),
        third,
        np.maximum(safe_second, global_scale),
    )

    local_scale = (
        0.62 * safe_best
        + 0.28 * safe_second
        + 0.10 * safe_third
    )
    local_scale = np.maximum(local_scale, eps)

    local_ratio = d / local_scale[:, None]
    global_ratio = d / global_scale

    local_quality = np.exp(
        -np.clip(local_ratio, 0.0, 60.0) / 0.72
    )
    global_quality = np.exp(
        -np.clip(global_ratio, 0.0, 60.0) / 1.55
    )

    # Opportunity cost: an edge is more valuable when both endpoints
    # have few alternatives of comparable quality.
    row_opportunity = np.maximum(safe_second - safe_best, 0.0)
    col_opportunity = row_opportunity.copy()

    row_urgency = np.clip(
        row_opportunity / np.maximum(safe_second, eps),
        0.0,
        1.0,
    )

    # Use a soft, bounded scarcity estimate instead of explicit global
    # matrix balancing.
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

    row_scarcity = 1.0 / np.sqrt(np.maximum(row_near_count, 1.0))
    col_scarcity = 1.0 / np.sqrt(np.maximum(col_near_count, 1.0))

    scarcity = (
        row_scarcity[:, None] + col_scarcity[None, :]
    )
    scarcity = scarcity / np.maximum(np.max(scarcity), 1.0)

    # Estimate the cost of consuming an edge relative to the best
    # alternatives at both endpoints.
    endpoint_regret = (
        0.5 * row_opportunity[:, None]
        + 0.5 * col_opportunity[None, :]
    )
    regret_scale = np.maximum(
        global_scale + 0.5 * (safe_second[:, None] + safe_second[None, :]),
        eps,
    )
    regret_factor = np.exp(
        -np.clip(endpoint_regret / regret_scale, 0.0, 8.0) / 2.4
    )

    reverse_valid = valid.T
    reverse_distance = np.where(reverse_valid, d.T, np.inf)
    reverse_quality = np.exp(
        -np.clip(reverse_distance / global_scale, 0.0, 60.0) / 1.25
    )

    mutual_quality = np.sqrt(
        np.maximum(local_quality, 0.0)
        * np.maximum(reverse_quality, 0.0)
    )
    mutual_quality = np.where(valid, mutual_quality, 0.0)

    # Long-term potential rewards edges that remain attractive after
    # accounting for endpoint alternatives, while staying subordinate
    # to the local distance signal.
    long_term = (
        0.55 * local_quality
        + 0.25 * global_quality
        + 0.20 * mutual_quality
    )
    long_term *= 0.88 + 0.12 * regret_factor

    score = (
        0.70 * local_quality
        + 0.14 * global_quality
        + 0.16 * long_term
    )

    # Bounded diversity correction. It helps under-covered endpoints
    # without equalizing all rows or columns.
    diversity_bonus = 1.0 + 0.18 * scarcity
    urgency_bonus = (
        1.0
        + 0.06 * row_urgency[:, None]
        + 0.06 * row_urgency[None, :]
    )

    score *= diversity_bonus
    score *= urgency_bonus
    score *= 0.93 + 0.07 * mutual_quality
    score = np.where(valid, np.maximum(score, 0.0), 0.0)

    # Preserve symmetry when the input represents an undirected TSP.
    symmetric_input = (
        np.array_equal(valid, valid.T)
        and np.allclose(
            np.where(valid, d, 0.0),
            np.where(valid.T, d.T, 0.0),
            rtol=1e-10,
            atol=1e-12,
        )
    )

    if symmetric_input:
        score = np.sqrt(
            np.maximum(score, 0.0)
            * np.maximum(score.T, 0.0)
        )

    # Small exploration prior, scaled by each node's available degree.
    degree = valid.sum(axis=1, keepdims=True).astype(float)
    uniform_prior = np.divide(
        valid.astype(float),
        np.maximum(degree, 1.0),
    )

    score = 0.94 * score + 0.06 * uniform_prior
    score = np.where(valid, np.maximum(score, 0.0), 0.0)
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
