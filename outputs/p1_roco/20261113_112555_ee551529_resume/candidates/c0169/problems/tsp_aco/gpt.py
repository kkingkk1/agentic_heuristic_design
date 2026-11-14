import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric distance- and neighborhood-aware TSP edge prior."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )
    symmetric_valid = directed_valid & directed_valid.T

    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    valid_values = distances[directed_valid]
    positive_values = valid_values[valid_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(valid_values)), 1.0)
    global_scale = max(global_scale, eps)

    symmetric_distances = np.where(
        symmetric_valid,
        0.5 * (distances + distances.T),
        np.inf,
    )
    np.fill_diagonal(symmetric_distances, np.inf)

    usable_neighbors = min(4, n - 1)
    ordered = np.sort(symmetric_distances, axis=1)
    nearest_distances = ordered[:, :usable_neighbors]

    local_scale = np.mean(
        np.where(
            np.isfinite(nearest_distances),
            nearest_distances,
            global_scale,
        ),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    normalized_distance = np.divide(
        symmetric_distances,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(symmetric_distances),
    )
    shortness = np.exp(-np.clip(normalized_distance, 0.0, 50.0))

    order = np.argsort(symmetric_distances, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[~np.isfinite(symmetric_distances)] = n

    rank_scarcity = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    nearest = np.argmin(symmetric_distances, axis=1)
    mutual_nearest = (
        np.arange(n)[:, None] == nearest[None, :]
    ) & (
        nearest[:, None] == np.arange(n)[None, :]
    )
    mutual_bonus = 1.0 + 0.30 * mutual_nearest.astype(float)

    isolation = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) / (
        np.maximum(symmetric_distances, 0.0) + global_scale
    )
    isolation = np.clip(
        np.nan_to_num(isolation, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )

    score = (
        shortness
        * (0.50 + 0.50 * rank_scarcity)
        * mutual_bonus
        * (1.0 + 0.12 * isolation)
    )

    score[~symmetric_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
