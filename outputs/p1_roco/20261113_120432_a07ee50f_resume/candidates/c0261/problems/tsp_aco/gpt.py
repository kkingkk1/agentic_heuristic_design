import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative symmetric edge-priority scores for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid_input = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(valid_input):
        return np.zeros((n, n), dtype=float)

    # Use the best valid direction for each pair, then enforce symmetry.
    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = safe[valid]
        global_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    # Stable endpoint-local scales from the nearest valid neighbors.
    usable = min(4, n - 1)
    sorted_distances = np.sort(safe, axis=1)
    nearest = sorted_distances[:, :usable]
    nearest_valid = np.isfinite(nearest)
    counts = np.sum(nearest_valid, axis=1)

    local_scale = np.sum(
        np.where(nearest_valid, nearest, 0.0),
        axis=1,
    ) / np.maximum(counts, 1)

    local_scale = np.where(counts > 0, local_scale, global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    scaled_distance = safe / np.maximum(pair_scale, eps)

    # Robust short-edge preference with a bounded tail.
    distance_score = 1.0 / (1.0 + np.clip(scaled_distance, 0.0, 1e6))

    # Compute endpoint ranks while excluding invalid and diagonal entries.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.full((n, n), float(n), dtype=float)
    col_rank = np.full((n, n), float(n), dtype=float)

    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    # Rank score is deliberately bounded and keeps alternatives available.
    row_rank_score = 1.0 / (1.0 + row_rank)
    col_rank_score = 1.0 / (1.0 + col_rank)
    rank_score = np.sqrt(row_rank_score * col_rank_score)

    # Small mutual-neighborhood bonus, capped to avoid lock-in.
    top_k = min(3, n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)
    top_neighbors[
        np.arange(n)[:, None],
        row_order[:, :top_k],
    ] = True
    top_neighbors &= valid

    mutual_top = top_neighbors & top_neighbors.T
    mutual_bonus = 1.0 + 0.12 * mutual_top.astype(float)

    # Keep a substantial baseline so stochastic sampling can explore.
    scores = (
        0.68 * distance_score
        + 0.32 * rank_score
    ) * mutual_bonus

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
