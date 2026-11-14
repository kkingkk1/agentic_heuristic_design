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
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(directed_valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distances = np.minimum(
        np.where(directed_valid, distances, np.inf),
        np.where(directed_valid.T, distances.T, np.inf),
    )

    valid = np.isfinite(symmetric_distances) & ~diagonal
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = symmetric_distances[
        valid & (symmetric_distances > 0.0)
    ]

    eps = np.finfo(float).eps
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = symmetric_distances[valid]
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    positive_distances = np.where(
        valid & (symmetric_distances > 0.0),
        symmetric_distances,
        np.inf,
    )
    sorted_positive = np.sort(positive_distances, axis=1)

    local_k = min(5, max(1, n - 1))
    local_neighbors = sorted_positive[:, :local_k]
    local_counts = np.sum(np.isfinite(local_neighbors), axis=1)
    local_sums = np.sum(
        np.where(np.isfinite(local_neighbors), local_neighbors, 0.0),
        axis=1,
    )

    local_scale = np.divide(
        local_sums,
        np.maximum(local_counts, 1),
        out=np.full(n, global_scale, dtype=float),
        where=local_counts > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_local = symmetric_distances / np.maximum(pair_scale, eps)
    normalized_global = symmetric_distances / global_scale

    local_score = np.exp(
        -np.power(np.clip(normalized_local, 0.0, 50.0), 1.25)
    )
    global_score = np.power(
        1.0 + np.clip(normalized_global, 0.0, 1e12),
        -0.35,
    )

    row_order = np.argsort(
        symmetric_distances,
        axis=1,
        kind="stable",
    )
    col_order = np.argsort(
        symmetric_distances,
        axis=0,
        kind="stable",
    )

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )
    rank_score = 0.55 + 0.45 * reciprocal_rank

    candidate_k = min(4, n - 1)
    row_candidates = np.zeros((n, n), dtype=bool)
    col_candidates = np.zeros((n, n), dtype=bool)

    row_candidates[
        np.arange(n)[:, None],
        row_order[:, :candidate_k],
    ] = True

    col_candidates[
        col_order[:candidate_k, :],
        np.arange(n)[None, :],
    ] = True

    mutual_candidates = (
        row_candidates
        & row_candidates.T
        & col_candidates
        & col_candidates.T
        & valid
    )

    nearest_row = np.argmin(symmetric_distances, axis=1)
    nearest_col = np.argmin(symmetric_distances, axis=0)

    mutual_nearest = (
        valid
        & (
            nearest_row[:, None]
            == np.arange(n)[None, :]
        )
        & (
            nearest_col[None, :]
            == np.arange(n)[:, None]
        )
    )

    compatibility_bonus = (
        1.0
        + 0.22 * mutual_candidates.astype(float)
        + 0.42 * mutual_nearest.astype(float)
    )

    endpoint_fit = np.sqrt(
        np.exp(
            -np.clip(
                symmetric_distances / local_scale[:, None],
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                symmetric_distances / local_scale[None, :],
                0.0,
                50.0,
            )
        )
    )

    isolation = np.clip(
        local_scale / global_scale - 1.0,
        0.0,
        2.0,
    )
    isolation_strength = np.sqrt(
        np.maximum(
            np.exp(
                -np.clip(
                    symmetric_distances / global_scale,
                    0.0,
                    50.0,
                )
            ),
            eps,
        )
    )
    isolation_bonus = 1.0 + 0.10 * (
        0.5 * (isolation[:, None] + isolation[None, :])
    ) * isolation_strength

    scores = (
        np.sqrt(local_score * global_score)
        * endpoint_fit
        * rank_score
        * compatibility_bonus
        * isolation_bonus
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
