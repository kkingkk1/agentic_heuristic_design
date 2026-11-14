import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    finite &= ~np.eye(n, dtype=bool)

    # Keep only edges available from both endpoints for an undirected TSP.
    valid = finite & finite.T
    if not np.any(valid):
        return scores

    d = np.full((n, n), np.inf, dtype=float)
    d[valid] = 0.5 * (distances[valid] + distances.T[valid])
    np.fill_diagonal(d, np.inf)

    finite_values = d[valid]
    positive = finite_values[finite_values > 0.0]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = max(float(np.max(finite_values)), 1.0)
    global_scale = max(global_scale, np.finfo(float).eps)

    # Stable local neighborhoods.
    order = np.argsort(d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(d, order, axis=1)

    best = sorted_d[:, 0]
    second = sorted_d[:, 1]
    local_scale = np.maximum(
        0.5 * (best + second),
        0.05 * global_scale,
    )
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    # Rank-based preference is less sensitive to heterogeneous distance units.
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank_preference = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    # Calibrated geometric proximity from both endpoints.
    normalized = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    proximity = np.exp(-np.clip(normalized, 0.0, 50.0))

    endpoint_quality = np.sqrt(
        np.exp(-np.clip(d / local_scale[:, None], 0.0, 50.0))
        * np.exp(-np.clip(d / local_scale[None, :], 0.0, 50.0))
    )

    # Preserve edges that are important to locally constrained endpoints,
    # but keep this correction bounded.
    regret = np.maximum(second - best, 0.0) / np.maximum(
        second,
        np.finfo(float).eps,
    )
    regret = np.clip(regret, 0.0, 1.0)
    regret_bonus = 1.0 + 0.25 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    mutual = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)
    mutual[rows, nearest] = True
    mutual &= mutual.T
    mutual_bonus = np.where(mutual, 1.18, 1.0)

    # Mild node-coverage correction. It discourages repeated concentration
    # around already highly accessible endpoints without rewarding long edges.
    local_mass = np.sum(
        np.where(valid, np.exp(-np.clip(d / local_scale[:, None], 0.0, 50.0)), 0.0),
        axis=1,
    )
    coverage = 1.0 / np.sqrt(np.maximum(local_mass, 1.0))
    coverage_pair = np.sqrt(
        coverage[:, None] * coverage[None, :]
    )
    coverage_pair /= max(float(np.max(coverage_pair)), np.finfo(float).eps)
    coverage_factor = 0.90 + 0.10 * coverage_pair

    scores = (
        proximity
        * (0.45 + 0.30 * rank_preference + 0.25 * endpoint_quality)
        * regret_bonus
        * mutual_bonus
        * coverage_factor
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    positive_scores = scores[scores > 0.0]
    if positive_scores.size:
        reference = float(np.quantile(positive_scores, 0.995))
        reference = max(reference, np.finfo(float).eps)
        scores = np.clip(scores / reference, 0.0, 1.0)

    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
