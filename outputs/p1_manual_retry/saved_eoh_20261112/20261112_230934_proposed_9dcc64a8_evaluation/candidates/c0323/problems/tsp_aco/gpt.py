import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-promise matrix for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(d) & (d >= 0.0)
    valid = finite & finite.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    # Robustly construct an undirected distance from both directions.
    forward = np.where(valid, d, np.inf)
    reverse = np.where(valid, d.T, np.inf)
    positive_pair = valid & (forward > 0.0) & (reverse > 0.0)

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid & ~positive_pair] = 0.0
    symmetric[positive_pair] = np.sqrt(
        forward[positive_pair] * reverse[positive_pair]
    )

    positive_values = symmetric[valid & (symmetric > 0.0)]
    reference = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    reference = max(reference, eps)

    work = np.where(valid, np.maximum(symmetric, eps * reference), np.inf)
    np.fill_diagonal(work, np.inf)

    degree = valid.sum(axis=1).astype(float)
    degree_safe = np.maximum(degree, 1.0)

    # Use row quantiles as competitive baselines rather than a single
    # nearest-neighbor scale.
    row_values = np.where(valid, work, np.nan)
    with np.errstate(all="ignore"):
        lower_quantile = np.nanquantile(row_values, 0.35, axis=1)
        upper_quantile = np.nanquantile(row_values, 0.70, axis=1)

    lower_quantile = np.where(
        np.isfinite(lower_quantile),
        lower_quantile,
        reference,
    )
    upper_quantile = np.where(
        np.isfinite(upper_quantile),
        upper_quantile,
        lower_quantile,
    )
    lower_quantile = np.maximum(lower_quantile, eps)
    upper_quantile = np.maximum(upper_quantile, lower_quantile)

    pair_baseline = np.sqrt(
        lower_quantile[:, None] * lower_quantile[None, :]
    )
    normalized_length = work / np.maximum(pair_baseline, eps)
    normalized_length = np.minimum(normalized_length, 1e6)

    # Measure whether a direct edge is substantially better than its best
    # feasible two-hop replacement.
    if n > 2:
        through_i = work[:, :, None]
        through_j = work[None, :, :]
        two_hop = through_i + through_j

        intermediary_valid = (
            valid[:, :, None]
            & valid[None, :, :]
        )

        node_ids = np.arange(n)
        intermediary_valid &= (
            node_ids[None, None, :] != node_ids[:, None, None]
        )
        intermediary_valid &= (
            node_ids[None, None, :] != node_ids[None, :, None]
        )

        two_hop = np.where(intermediary_valid, two_hop, np.inf)
        best_two_hop = np.min(two_hop, axis=2)
    else:
        best_two_hop = np.full((n, n), np.inf, dtype=float)

    finite_detour = np.isfinite(best_two_hop)
    relative_detour = np.where(
        finite_detour,
        best_two_hop / np.maximum(work, eps),
        1.0,
    )
    direct_saving = np.maximum(relative_detour - 1.0, 0.0)
    direct_saving = np.minimum(direct_saving, 20.0)

    # Stable edge ranks among each endpoint's feasible alternatives.
    order = np.argsort(work, axis=1, kind="stable")
    positions = np.empty((n, n), dtype=float)
    positions[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    normalized_rank = positions / degree_safe[:, None]
    normalized_rank = np.where(valid, normalized_rank, 1.0)

    # A high value means the edge is competitive for both endpoints.
    bilateral_competition = np.sqrt(
        np.maximum(1.0 - normalized_rank, 0.0)
        * np.maximum(1.0 - normalized_rank.T, 0.0)
    )
    bilateral_competition = np.maximum(bilateral_competition, eps)

    # Penalize edges that consume a large share of either endpoint's
    # available distance budget, while retaining bridge edges through saving.
    endpoint_budget = np.maximum(
        upper_quantile[:, None],
        upper_quantile[None, :],
    )
    budget_overrun = np.maximum(work / np.maximum(endpoint_budget, eps) - 1.0, 0.0)
    budget_overrun = np.minimum(budget_overrun, 20.0)

    # Soft degree correction: distribute probability mass away from nodes
    # whose incident edges are all highly competitive.
    competitive_mass = np.sum(
        np.exp(-np.minimum(work / np.maximum(lower_quantile[:, None], eps), 40.0))
        * valid,
        axis=1,
    )
    competitive_mass = np.maximum(competitive_mass, 1.0)

    hub_correction = 1.0 / np.sqrt(
        competitive_mass[:, None] * competitive_mass[None, :]
    )
    hub_correction = np.maximum(hub_correction, eps)

    # Smooth preference for direct short connections.
    shortness = np.exp(
        -np.minimum(normalized_length, 40.0)
    )

    log_score = (
        1.35 * np.log(np.maximum(shortness, eps))
        + 1.10 * np.log(bilateral_competition)
        + 0.95 * np.log1p(direct_saving)
        + 0.24 * np.log(hub_correction)
        - 0.30 * budget_overrun
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - maximum, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
