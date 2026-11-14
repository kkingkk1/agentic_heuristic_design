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

    finite = np.isfinite(distances)
    nonnegative = distances >= 0.0
    valid_input = finite & nonnegative
    valid_input &= ~np.eye(n, dtype=bool)

    if not np.any(valid_input):
        return scores

    # Symmetrize the available distances for an undirected TSP.
    d = np.where(valid_input, distances, np.inf)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)
    np.fill_diagonal(d, np.inf)

    if not np.any(valid):
        return scores

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.max(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    # Replace unavailable edges with a finite, strongly penalized distance.
    maximum_distance = float(np.max(finite_values))
    penalty_distance = max(maximum_distance, global_scale) * 4.0
    penalty_distance += global_scale
    work = np.where(valid, d, penalty_distance)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    sorted_distances = np.sort(work, axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1] if n > 2 else best

    local_scale = 0.5 * (best + second)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative_distance = work / np.maximum(pair_scale, eps)

    # Distance attractiveness with a smooth long-edge suppression.
    distance_term = np.exp(-np.clip(relative_distance, 0.0, 60.0))
    long_edge_penalty = np.exp(
        -0.45 * np.maximum(relative_distance - 1.0, 0.0)
    )

    # Bidirectional neighborhood preference.
    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_strength = np.exp(-ranks / rank_scale)
    bidirectional_rank = np.sqrt(rank_strength * rank_strength.T)

    # Edges selected by both endpoints receive a moderate bonus.
    nearest = order[:, 0]
    mutual_nearest = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.30, 1.0)

    # Preserve edges that are important to locally constrained endpoints.
    regret = np.maximum(second - best, 0.0) / np.maximum(second, eps)
    regret = np.clip(regret, 0.0, 1.0)
    regret_bonus = 1.0 + 0.45 * (
        regret[:, None] + regret[None, :]
    )

    # Endpoint accessibility adds a bounded two-sided structural signal.
    row_accessibility = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 60.0)
    )
    col_accessibility = np.exp(
        -np.clip(work / local_scale[None, :], 0.0, 60.0)
    )
    accessibility = np.sqrt(row_accessibility * col_accessibility)

    scores = (
        distance_term
        * (0.45 + 0.25 * bidirectional_rank + 0.30 * accessibility)
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
