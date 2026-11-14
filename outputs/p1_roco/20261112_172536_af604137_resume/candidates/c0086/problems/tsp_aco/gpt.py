import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    valid &= ~np.eye(n, dtype=bool)

    positive_values = distances[valid]
    if positive_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive_values))
    d = np.where(valid, distances, fallback)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps

    row_sorted = np.sort(d, axis=1)
    col_sorted = np.sort(d, axis=0)

    row_best = np.maximum(row_sorted[:, 0], eps)
    row_second = np.maximum(row_sorted[:, 1], row_best)
    col_best = np.maximum(col_sorted[0, :], eps)
    col_second = np.maximum(col_sorted[1, :], col_best)

    row_ratio = d / row_best[:, None]
    col_ratio = d / col_best[None, :]

    two_sided_ratio = np.sqrt(row_ratio * col_ratio)

    # Smooth local proximity avoids the excessive sharpness of a pure inverse-distance score.
    proximity = 1.0 / (1.0 + two_sided_ratio)
    visibility = 1.0 / np.maximum(two_sided_ratio, eps)

    # Regret measures how much worse an edge is than the best option at either endpoint.
    row_regret = np.maximum(d - row_best[:, None], 0.0) / (
        row_second[:, None] - row_best[:, None] + eps
    )
    col_regret = np.maximum(d - col_best[None, :], 0.0) / (
        col_second[None, :] - col_best[None, :] + eps
    )

    endpoint_fit = np.exp(
        -0.5 * (np.sqrt(row_regret) + np.sqrt(col_regret))
    )

    # Give a bounded bonus to edges that are important because their endpoints
    # have unusually poor alternatives.
    row_criticality = np.clip(
        (row_second - row_best) / (row_second + eps), 0.0, 1.0
    )
    col_criticality = np.clip(
        (col_second - col_best) / (col_second + eps), 0.0, 1.0
    )

    critical_bonus = 1.0 + 0.25 * (
        row_criticality[:, None] * np.exp(-row_regret)
        + col_criticality[None, :] * np.exp(-col_regret)
    )

    # Weak long-term continuation potential: i -> j is preferred when j has
    # a promising outgoing continuation and i has a promising incoming option.
    finite_values = distances[valid]
    global_scale = max(float(np.median(finite_values)), eps)

    continuation_scale = np.sqrt(
        row_best[None, :] * col_best[:, None]
    )
    continuation = 1.0 / (1.0 + continuation_scale / global_scale)

    # Deliberately bounded tie-breaker; local evidence remains dominant.
    long_term_factor = 0.97 + 0.06 * continuation

    score = (
        visibility
        * proximity
        * endpoint_fit
        * critical_bonus
        * long_term_factor
    )

    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
