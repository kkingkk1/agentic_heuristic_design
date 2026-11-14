import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
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

    symmetric_d = 0.5 * (d + d.T)
    symmetric_d = np.maximum(symmetric_d, 0.0)

    positive_values = symmetric_d[valid & (symmetric_d > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
        lower_scale = float(np.percentile(positive_values, 25.0))
        upper_scale = float(np.percentile(positive_values, 75.0))
    else:
        global_scale = 1.0
        lower_scale = 1.0
        upper_scale = 1.0

    global_scale = max(global_scale, eps)
    lower_scale = max(lower_scale, eps)
    upper_scale = max(upper_scale, global_scale)

    work = np.where(valid, symmetric_d, np.inf)

    row_values = np.where(valid, symmetric_d, np.nan)
    row_positive = np.where(valid & (symmetric_d > 0.0), symmetric_d, np.nan)

    row_temperature = np.nanmedian(row_positive, axis=1)
    row_temperature = np.where(
        np.isfinite(row_temperature),
        row_temperature,
        global_scale,
    )
    row_temperature = np.maximum(
        row_temperature,
        0.35 * lower_scale,
    )

    # A soft endpoint preference, based on probability-like competition
    # rather than explicit nearest-neighbor ranks.
    with np.errstate(over="ignore", invalid="ignore"):
        endpoint_kernel = np.exp(
            -work / row_temperature[:, None]
        )

    endpoint_kernel = np.where(valid, endpoint_kernel, 0.0)
    endpoint_mass = np.sum(endpoint_kernel, axis=1)
    endpoint_mass = np.maximum(endpoint_mass, eps)

    endpoint_choice = endpoint_kernel / endpoint_mass[:, None]
    mutual_choice = np.sqrt(
        np.maximum(endpoint_choice, eps)
        * np.maximum(endpoint_choice.T, eps)
    )

    # Estimate how difficult each edge is to replace by a two-hop route.
    # The three best valid detours provide a robust alternative-cost summary.
    best1 = np.full((n, n), np.inf, dtype=float)
    best2 = np.full((n, n), np.inf, dtype=float)
    best3 = np.full((n, n), np.inf, dtype=float)

    if n >= 3:
        for k in range(n):
            through_k_valid = (
                valid[:, k, None]
                & valid[k, None, :]
            )
            through_k = (
                symmetric_d[:, k, None]
                + symmetric_d[k, None, :]
            )
            candidate = np.where(through_k_valid, through_k, np.inf)

            improve = candidate < best1
            old1 = best1.copy()
            old2 = best2.copy()

            best1 = np.where(improve, candidate, best1)
            best2 = np.where(
                improve,
                old1,
                np.minimum(best2, candidate),
            )
            best3 = np.where(
                improve,
                old2,
                np.minimum(best3, candidate),
            )

    finite_best1 = np.isfinite(best1)
    finite_best2 = np.isfinite(best2)
    finite_best3 = np.isfinite(best3)

    detour_sum = np.zeros((n, n), dtype=float)
    detour_count = (
        finite_best1.astype(float)
        + finite_best2.astype(float)
        + finite_best3.astype(float)
    )

    detour_sum += np.where(finite_best1, best1, 0.0)
    detour_sum += np.where(finite_best2, best2, 0.0)
    detour_sum += np.where(finite_best3, best3, 0.0)

    fallback_detour = symmetric_d + upper_scale
    alternative_cost = np.divide(
        detour_sum,
        np.maximum(detour_count, 1.0),
    )
    alternative_cost = np.where(
        detour_count > 0.0,
        alternative_cost,
        fallback_detour,
    )

    direct_cost = np.where(valid, symmetric_d, 0.0)

    # Positive margin means that the direct edge avoids a costly detour.
    detour_margin = (
        alternative_cost - direct_cost
    ) / global_scale
    detour_margin = np.clip(detour_margin, -8.0, 8.0)

    # Smoothly reward useful direct shortcuts while avoiding hard thresholds.
    shortcut_value = 1.0 / (
        1.0 + np.exp(-1.35 * detour_margin)
    )

    # Penalize edges that are globally extreme, but retain a nonzero floor
    # so that necessary long connections are not removed entirely.
    global_relative = direct_cost / upper_scale
    global_preference = np.exp(
        -0.22 * np.maximum(global_relative - 1.0, 0.0)
    )
    global_preference = 0.30 + 0.70 * global_preference

    # Edges whose two-hop replacement is much cheaper are less promising.
    replacement_ratio = np.divide(
        direct_cost,
        np.maximum(alternative_cost, eps),
    )
    replacement_penalty = 1.0 / (
        1.0 + np.exp(-2.0 * (replacement_ratio - 0.85))
    )
    replacement_penalty = 0.25 + 0.75 * replacement_penalty

    # Slightly favor edges that are simultaneously competitive for both ends.
    competition_factor = np.sqrt(
        np.maximum(mutual_choice, eps)
    )
    competition_factor = np.power(competition_factor, 0.72)

    with np.errstate(over="ignore", invalid="ignore"):
        score = (
            np.power(np.maximum(shortcut_value, eps), 1.35)
            * np.power(np.maximum(competition_factor, eps), 1.10)
            * np.power(np.maximum(global_preference, eps), 0.65)
            * np.power(np.maximum(replacement_penalty, eps), 0.85)
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
