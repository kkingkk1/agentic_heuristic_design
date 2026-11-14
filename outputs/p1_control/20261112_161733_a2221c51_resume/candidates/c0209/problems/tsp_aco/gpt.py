import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
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

    d = 0.5 * (d + d.T)

    eps = np.finfo(float).eps
    positive = d[valid & (d > 0.0)]

    if positive.size:
        global_scale = max(float(np.median(positive)), eps)
        fallback = max(float(np.max(positive)), global_scale)
    else:
        global_scale = 1.0
        fallback = 1.0

    safe = np.where(valid, np.maximum(d, eps), np.inf)
    np.fill_diagonal(safe, np.inf)

    best_detour = np.full((n, n), np.inf, dtype=float)
    alternative_mass = np.zeros((n, n), dtype=float)

    tolerance = max(0.35 * global_scale, eps)

    for k in range(n):
        two_hop = safe[:, k, None] + safe[None, k, :]
        usable = valid & np.isfinite(two_hop)

        best_detour = np.minimum(
            best_detour,
            np.where(usable, two_hop, np.inf),
        )

        excess = np.maximum(two_hop - safe, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            closeness = np.exp(-excess / tolerance)

        alternative_mass += np.where(usable, closeness, 0.0)

    if n > 2:
        redundancy = alternative_mass / float(n - 2)
    else:
        redundancy = np.zeros((n, n), dtype=float)

    has_detour = valid & np.isfinite(best_detour)
    ratio = np.ones((n, n), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio[has_detour] = best_detour[has_detour] / safe[has_detour]

    log_ratio = np.zeros((n, n), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio[has_detour] = np.log(
            np.maximum(ratio[has_detour], eps)
        )

    clipped_log_ratio = np.clip(log_ratio, -8.0, 8.0)
    with np.errstate(over="ignore"):
        detour_resistance = 1.0 / (
            1.0 + np.exp(-1.8 * clipped_log_ratio)
        )

    relative_distance = np.minimum(
        np.maximum(safe / global_scale, 0.0),
        1e6,
    )

    log_score = (
        -0.30 * relative_distance
        + 1.35 * np.log(np.maximum(detour_resistance, eps))
        - 0.95 * redundancy
    )

    log_score = np.where(valid, log_score, -np.inf)

    finite_scores = np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(log_score[finite_scores]))

    with np.errstate(over="ignore", invalid="ignore"):
        score = np.exp(np.clip(log_score - shift, -745.0, 0.0))

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
