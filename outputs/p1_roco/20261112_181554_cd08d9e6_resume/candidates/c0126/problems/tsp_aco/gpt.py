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

    valid = np.isfinite(distances) & (distances >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return scores

    d = np.where(valid, distances, np.inf)
    d = 0.5 * (d + d.T)

    valid = np.isfinite(d)
    np.fill_diagonal(valid, False)
    np.fill_diagonal(d, np.inf)

    if not np.any(valid):
        return scores

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else max(float(np.max(finite_values)), 1.0)
    )
    scale = max(scale, np.finfo(float).eps)

    penalty = max(float(np.max(finite_values)), scale) * 3.5 + scale
    work = np.where(valid, d, penalty)

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]

    sorted_distances = np.sort(work, axis=1)
    best = sorted_distances[:, 0]
    second = sorted_distances[:, 1]

    local_scale = np.maximum(
        0.5 * (best + second),
        np.finfo(float).eps,
    )

    relative_distance = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    distance_term = np.exp(-np.clip(0.95 * relative_distance, 0.0, 60.0))

    rank_term = 1.0 / np.sqrt(
        (1.0 + 0.85 * ranks) * (1.0 + 0.85 * ranks.T)
    )

    row_quality = np.exp(
        -np.clip(work / local_scale[:, None], 0.0, 60.0)
    )
    col_quality = np.exp(
        -np.clip(work / local_scale[None, :], 0.0, 60.0)
    )
    accessibility = np.sqrt(row_quality * col_quality)

    regret = np.maximum(second - best, 0.0) / np.maximum(
        second,
        np.finfo(float).eps,
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.28 * (
        regret[:, None] + regret[None, :]
    )

    nearest = order[:, 0]
    mutual = (
        (nearest[:, None] == np.arange(n)[None, :])
        & (nearest[None, :] == np.arange(n)[:, None])
    )
    mutual_bonus = np.where(mutual, 1.18, 1.0)

    normalized_distance = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    outlier_penalty = 1.0 / (
        1.0
        + np.maximum(0.0, normalized_distance - 1.8)
    )

    scores = (
        distance_term
        * (0.46 + 0.30 * rank_term + 0.24 * accessibility)
        * regret_bonus
        * mutual_bonus
        * outlier_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
