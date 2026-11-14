import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    positive = distances[finite & (distances > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = max(float(np.median(distances[finite])), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    forward = np.where(finite, distances, np.inf)
    backward = forward.T

    # Use the best available direction while preserving a symmetric score.
    safe = np.minimum(forward, backward)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, safe, np.inf)

    usable = min(max(3, int(np.sqrt(n)) + 1), n - 1)
    sorted_distances = np.sort(safe, axis=1)

    local_block = sorted_distances[:, :usable]
    local_block = np.where(np.isfinite(local_block), local_block, global_scale)

    local_scale = np.mean(local_block, axis=1)
    local_scale = np.maximum(local_scale, eps)

    # Robustly limit the influence of highly isolated endpoints.
    local_scale = np.minimum(local_scale, 3.0 * global_scale)

    endpoint_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )

    pair_distance = safe / endpoint_scale
    row_distance = safe / np.maximum(local_scale[:, None], eps)
    col_distance = safe / np.maximum(local_scale[None, :], eps)

    # Continuous short-edge preference with stronger attenuation for long edges.
    distance_term = np.exp(
        -np.clip(pair_distance ** 1.35, 0.0, 60.0)
    )

    endpoint_term = np.exp(
        -0.5
        * np.clip(row_distance ** 1.20 + col_distance ** 1.20, 0.0, 60.0)
    )

    # Stable reciprocal ranks provide a softer alternative to hard top-k rules.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)[:, None]

    rank_term = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    # Reward edges that are preferred by both endpoints.
    nearest_row = row_order[:, 0]
    nearest_col = col_order[0, :]

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    )

    top_k = min(max(2, int(np.sqrt(n))), n - 1)
    row_candidates = np.zeros((n, n), dtype=bool)
    row_candidates[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True

    mutual_candidates = row_candidates & row_candidates.T & valid

    # Mild density correction favors edges useful to both local neighborhoods.
    density_term = np.sqrt(
        np.maximum(local_scale[:, None], eps)
        / np.maximum(local_scale[None, :], eps)
    )
    density_term = 0.5 * (density_term + density_term.T)
    density_term = np.clip(density_term, 0.85, 1.15)

    scores = (
        distance_term
        * endpoint_term
        * (0.42 + 0.58 * rank_term)
        * (1.0 + 0.72 * mutual_nearest.astype(float))
        * (1.0 + 0.24 * mutual_candidates.astype(float))
        * density_term
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
