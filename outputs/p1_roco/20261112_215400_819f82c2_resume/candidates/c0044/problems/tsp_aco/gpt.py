import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

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

    # Robust endpoint-local scale from the nearest few usable edges.
    sorted_distances = np.sort(safe, axis=1)
    usable = min(3, n - 1)
    local_scale = np.mean(sorted_distances[:, :usable], axis=1)
    local_scale = np.where(np.isfinite(local_scale), local_scale, global_scale)
    local_scale = np.maximum(local_scale, eps)

    # Two-sided local affinity.
    row_affinity = np.exp(
        -np.clip(safe / local_scale[:, None], 0.0, 50.0)
    )
    col_affinity = np.exp(
        -np.clip(safe / local_scale[None, :], 0.0, 50.0)
    )
    affinity = np.sqrt(row_affinity * col_affinity)

    # Reciprocal rank scarcity: strong edges from both endpoints score higher.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)[None, :]
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    # Mutual-nearest-neighbor reinforcement.
    nearest_row = np.argmin(safe, axis=1)
    nearest_col = np.argmin(safe, axis=0)
    mutual = (
        valid
        & (nearest_col[nearest_row] == np.arange(n))
        [ :, None]
    )
    mutual = mutual.astype(float)

    # Mild pressure toward edges useful to locally isolated endpoints.
    isolation = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    isolation_bonus = 1.0 + 0.15 * np.clip(
        isolation / (safe + global_scale),
        0.0,
        1.0,
    )

    score = (
        affinity
        * (0.60 + 0.40 * rank_score)
        * (1.0 + 0.25 * mutual)
        * isolation_bonus
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
