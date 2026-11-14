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
    finite = np.isfinite(distances)
    nonnegative = distances >= 0.0
    valid_input = finite & nonnegative & ~diagonal

    if not np.any(valid_input):
        return np.zeros((n, n), dtype=float)

    # TSP edges are treated as undirected.  The smaller available direction
    # is retained when the supplied matrix is asymmetric.
    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = safe[valid & (safe > 0.0)]

    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = safe[valid]
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    # Use nearest-neighbor distances to estimate endpoint-specific density.
    masked = np.where(valid, safe, np.nan)
    nearest = np.nanmin(masked, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    local_k = min(5, n - 1)
    nearest_rows = np.sort(np.where(valid, safe, np.inf), axis=1)[:, :local_k]
    local_scale = np.mean(nearest_rows, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    pair_scale = np.maximum(pair_scale, eps)

    positive_safe = np.maximum(safe, eps)

    # A power-law term preserves separation among short edges better than
    # multiplying several saturating exponentials.
    global_term = np.power(
        global_scale / np.maximum(positive_safe, global_scale * eps),
        1.15,
    )
    local_term = np.power(
        pair_scale / positive_safe,
        0.85,
    )

    # Penalize edges that are long relative to either endpoint's neighborhood.
    endpoint_ratio = 0.5 * (
        safe / local_scale[:, None]
        + safe / local_scale[None, :]
    )
    locality_term = 1.0 / (1.0 + np.power(np.maximum(endpoint_ratio, 0.0), 1.35))

    row_order = np.argsort(
        np.where(valid, safe, np.inf),
        axis=1,
        kind="stable",
    )
    col_order = np.argsort(
        np.where(valid, safe, np.inf),
        axis=0,
        kind="stable",
    )

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)[:, None]

    rank_term = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )

    # Continuous mutual-neighborhood agreement gives more information than
    # a single nearest-neighbor indicator while remaining bounded.
    k = min(4, n - 1)
    row_top = np.zeros((n, n), dtype=bool)
    col_top = np.zeros((n, n), dtype=bool)
    row_top[np.arange(n)[:, None], row_order[:, :k]] = True
    col_top[col_order[:, :k], np.arange(n)[None, :]] = True

    mutual_top = (row_top & row_top.T & valid).astype(float)
    mutual_directional = (row_top & col_top & valid).astype(float)

    nearest_row = np.argmin(np.where(valid, safe, np.inf), axis=1)
    nearest_col = np.argmin(np.where(valid, safe, np.inf), axis=0)

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    ).astype(float)

    neighborhood_term = (
        1.0
        + 0.22 * mutual_top
        + 0.16 * mutual_directional
        + 0.34 * mutual_nearest
    )

    # A small density compensation helps isolated endpoints without allowing
    # it to dominate distance and neighborhood evidence.
    density_ratio = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    ) / (global_scale + positive_safe)

    isolation_term = 1.0 + 0.08 * np.clip(density_ratio, 0.0, 1.0)

    scores = (
        global_term
        * local_term
        * locality_term
        * (0.42 + 0.58 * rank_term)
        * neighborhood_term
        * isolation_term
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
