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
    valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)
    global_scale = max(global_scale, eps)

    safe = np.where(valid, distances, np.inf)

    # Use a symmetric working distance matrix while preserving edge validity.
    safe = 0.5 * (safe + safe.T)
    symmetric_valid = valid & valid.T
    safe[~symmetric_valid] = np.inf
    np.fill_diagonal(safe, np.inf)

    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    # Robust local scale from the nearest few usable edges.
    usable = min(4, n - 1)
    sorted_distances = np.sort(safe, axis=1)
    local_values = sorted_distances[:, :usable]
    local_scale = np.mean(
        np.where(np.isfinite(local_values), local_values, global_scale),
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

    # Bidirectional relative shortness.
    relative_shortness = np.exp(
        -np.clip(safe / pair_scale, 0.0, 50.0)
    )

    # Reciprocal rank scarcity from both endpoints.
    row_order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / np.sqrt((1.0 + ranks) * (1.0 + ranks.T))

    # Mutual-nearest-neighbor reinforcement.
    nearest = np.argmin(safe, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i, j in enumerate(nearest):
        if np.isfinite(safe[i, j]) and nearest[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Mildly favor edges incident to locally isolated nodes.
    isolation_ratio = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) / (
        safe + 0.85 * global_scale
    )
    isolation_ratio = np.clip(
        np.nan_to_num(isolation_ratio, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )

    score = (
        relative_shortness
        * (0.50 + 0.50 * rank_score)
        * (1.0 + 0.30 * mutual)
        * (1.0 + 0.18 * isolation_ratio)
    )

    score[~symmetric_valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
