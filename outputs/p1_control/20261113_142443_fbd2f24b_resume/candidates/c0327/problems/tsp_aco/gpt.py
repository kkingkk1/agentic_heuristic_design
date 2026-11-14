import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structural edge-promise indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    d = 0.5 * (d + d.T)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive_values = d[valid & (d > 0.0)]
    if positive_values.size:
        global_scale = max(float(np.median(positive_values)), eps)
    else:
        global_scale = 1.0

    work = np.where(valid, d, np.inf)

    valid_count = valid.sum(axis=1).astype(float)
    row_sum = np.sum(np.where(valid, d, 0.0), axis=1)
    row_mean = np.divide(
        row_sum,
        np.maximum(valid_count, 1.0),
        out=np.full(n, global_scale, dtype=float),
        where=valid_count > 0,
    )

    centered = np.where(valid, d - row_mean[:, None], 0.0)
    row_variance = np.divide(
        np.sum(centered * centered, axis=1),
        np.maximum(valid_count, 1.0),
        out=np.full(n, global_scale * global_scale, dtype=float),
        where=valid_count > 0,
    )
    row_scale = np.sqrt(np.maximum(row_variance, eps))
    row_scale = np.maximum(row_scale, 0.25 * global_scale)

    # A continuous cost signal based on endpoint-specific standardization.
    endpoint_center = 0.5 * (
        row_mean[:, None] + row_mean[None, :]
    )
    endpoint_scale = np.sqrt(
        row_scale[:, None] * row_scale[None, :]
    )
    standardized_cost = np.divide(
        work - endpoint_center,
        endpoint_scale + eps,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(work),
    )
    standardized_cost = np.clip(standardized_cost, -20.0, 20.0)

    with np.errstate(over="ignore", invalid="ignore"):
        cost_preference = 1.0 / (
            1.0 + np.exp(0.85 * standardized_cost)
        )

    # Compute the cheapest valid two-hop replacement for every edge.
    best_detour = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        through_k = work[:, k, None] + work[None, k, :]
        through_k = np.where(
            np.isfinite(through_k),
            through_k,
            np.inf,
        )
        best_detour = np.minimum(best_detour, through_k)

    # Direct-edge savings over the best two-hop alternative.
    finite_detour = np.isfinite(best_detour) & valid
    detour_margin = np.ones((n, n), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        finite_margin = (
            best_detour - work
        ) / (
            best_detour + work + eps
        )

    detour_margin[finite_detour] = np.clip(
        finite_margin[finite_detour],
        -1.0,
        1.0,
    )

    # Smoothly reward edges that are structurally difficult to replace.
    with np.errstate(over="ignore", invalid="ignore"):
        bridge_preference = 0.35 + 0.65 / (
            1.0 + np.exp(-5.0 * detour_margin)
        )

    # Mild compatibility term prevents extreme endpoint regimes from
    # dominating solely because of the two-hop calculation.
    mean_difference = np.abs(
        row_mean[:, None] - row_mean[None, :]
    )
    mean_sum = row_mean[:, None] + row_mean[None, :] + eps
    compatibility = 0.82 + 0.18 / (
        1.0 + mean_difference / mean_sum
    )

    score = (
        cost_preference
        * bridge_preference
        * compatibility
    )

    score = np.where(valid, score, 0.0)
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
