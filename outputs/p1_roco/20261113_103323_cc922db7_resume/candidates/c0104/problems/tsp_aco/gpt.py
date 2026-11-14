import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-priority scores for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid = finite_nonnegative & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, distances, np.inf)

    positive = distances[valid & (distances > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = distances[valid]
        global_scale = float(np.median(finite_values)) if finite_values.size else 1.0
    global_scale = max(global_scale, eps)

    # Use the mean of the nearest few finite neighbors as a robust local scale.
    usable = min(3, n - 1)
    nearest_values = np.sort(safe, axis=1)[:, :usable]
    local_scale = np.mean(nearest_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    # Symmetric endpoint-adaptive distance desirability.
    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = safe / np.maximum(endpoint_scale, eps)
    normalized_distance = np.clip(normalized_distance, 0.0, 60.0)
    local_affinity = np.exp(-normalized_distance)

    global_distance = np.clip(safe / global_scale, 0.0, 60.0)
    global_affinity = 1.0 / (1.0 + global_distance ** 2)

    # Reciprocal rank scarcity.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    positions = np.arange(n, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = positions[None, :]
    col_rank[col_order, np.arange(n)[None, :]] = positions[:, None]

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    # Mutual nearest and mutual top-two neighborhood indicators.
    nearest_row = np.argmin(safe, axis=1)
    nearest_col = np.argmin(safe, axis=0)

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    )

    top_k = min(2, n - 1)
    row_top_two = np.zeros((n, n), dtype=bool)
    col_top_two = np.zeros((n, n), dtype=bool)

    row_top_two[np.arange(n)[:, None], row_order[:, :top_k]] = True
    col_top_two[col_order[:top_k, :], np.arange(n)[None, :]] = True

    mutual_top_two = row_top_two & col_top_two & valid

    # Reward edges incident to locally isolated endpoints without dominating distance.
    isolation_ratio = endpoint_scale / (safe + global_scale)
    isolation_ratio = np.clip(isolation_ratio, 0.0, 1.0)
    isolation_bonus = 1.0 + 0.20 * isolation_ratio

    score = (
        local_affinity
        * (0.55 + 0.45 * reciprocal_rank)
        * (0.75 + 0.25 * global_affinity)
        * (1.0 + 0.30 * mutual_nearest.astype(float))
        * (1.0 + 0.10 * mutual_top_two.astype(float))
        * isolation_bonus
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
