import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)

    d = np.nan_to_num(
        distances,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d) & off_diagonal
    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    positive = finite & (d > 0)
    positive_values = d[positive]

    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    finite_max = float(np.max(d[finite]))
    replacement = max(finite_max, global_scale)
    safe_d = np.where(finite, d, replacement)
    safe_d = np.maximum(safe_d, np.finfo(float).eps)

    sorted_distances = np.sort(safe_d, axis=1)
    local_index = min(max(1, int(np.ceil(np.sqrt(n))) - 1), n - 2)
    local_scale = sorted_distances[:, local_index]
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    normalized_distance = safe_d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    local_attraction = np.exp(-np.minimum(normalized_distance, 50.0))

    inverse_distance = 1.0 / (safe_d + 0.05 * global_scale)
    inverse_distance /= max(
        float(np.max(inverse_distance[finite])),
        np.finfo(float).eps,
    )

    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_preference = np.exp(
        -(ranks + ranks.T) / max(np.sqrt(n), 1.0)
    )

    nearest_k = min(n - 1, max(1, int(np.ceil(np.sqrt(n)))))
    nearest = np.zeros((n, n), dtype=float)
    nearest[np.arange(n)[:, None], order[:, :nearest_k]] = 1.0
    mutual_support = nearest * nearest.T
    one_sided_support = 0.5 * (nearest + nearest.T)
    neighborhood_bonus = 1.0 + 0.60 * mutual_support + 0.20 * one_sided_support

    competition = np.sum(local_attraction * finite, axis=1)
    competition = np.maximum(competition, np.finfo(float).eps)
    diversity = 1.0 / np.sqrt(
        competition[:, None] * competition[None, :]
    )

    score = (
        np.maximum(inverse_distance, 1e-12) ** 0.35
        * np.maximum(local_attraction, 1e-12) ** 0.35
        * np.maximum(rank_preference, 1e-12) ** 0.15
        * np.maximum(neighborhood_bonus, 1e-12) ** 0.10
        * np.maximum(diversity, 1e-12) ** 0.05
    )

    score[~finite] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
