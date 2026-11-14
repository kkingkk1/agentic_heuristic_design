import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distances) & (distances > 0) & off_diagonal

    finite_values = distances[valid]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.median(finite_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    d = np.where(valid, distances, fallback)
    np.fill_diagonal(d, np.inf)

    # Directed two-sided local normalization.
    k = min(6, n - 1)

    row_neighbors = np.partition(d, kth=k - 1, axis=1)[:, :k]
    col_neighbors = np.partition(d, kth=k - 1, axis=0)[:k, :]

    row_scale = np.median(row_neighbors, axis=1)
    col_scale = np.median(col_neighbors, axis=0)

    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        fallback,
    )
    col_scale = np.where(
        np.isfinite(col_scale) & (col_scale > 0),
        col_scale,
        fallback,
    )

    eps = np.finfo(float).eps
    pair_scale = np.sqrt(row_scale[:, None] * col_scale[None, :])
    relative_distance = d / np.maximum(pair_scale, eps)

    # Primary signal: short edges with strong two-sided rank.
    row_order = np.argsort(d, axis=1)
    col_order = np.argsort(d, axis=0)

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)

    rank_scale = max(1.0, np.sqrt(float(n)) * 0.9)
    row_rank_strength = np.exp(-row_rank / rank_scale)
    col_rank_strength = np.exp(-col_rank / rank_scale)
    two_sided_rank = np.sqrt(row_rank_strength * col_rank_strength)

    proximity = np.exp(-0.85 * relative_distance)
    long_edge_penalty = np.exp(
        -0.70 * np.maximum(relative_distance - 1.0, 0.0)
    )

    base_score = (
        (0.55 + 0.45 * proximity)
        * (0.58 + 0.42 * two_sided_rank)
        * long_edge_penalty
    )

    # Endpoint regret: nodes with a large gap between their best and
    # second-best alternatives receive a small bounded priority bonus.
    row_best = row_neighbors[:, 0]
    row_second = row_neighbors[:, min(1, k - 1)]
    col_best = col_neighbors[0, :]
    col_second = col_neighbors[min(1, k - 1), :]

    row_regret = np.clip(
        (row_second - row_best) / np.maximum(row_second, eps),
        0.0,
        1.0,
    )
    col_regret = np.clip(
        (col_second - col_best) / np.maximum(col_second, eps),
        0.0,
        1.0,
    )

    endpoint_bonus = (
        1.0
        + 0.18 * row_regret[:, None]
        + 0.18 * col_regret[None, :]
    )

    # Residual degree-feasibility proxy. After choosing (i, j), estimate
    # whether i and j still have affordable alternatives for their other
    # tour incidence.
    row_residual = np.partition(d, kth=1, axis=1)[:, 1]
    col_residual = np.partition(d, kth=1, axis=0)[1, :]

    residual_cost = (
        row_residual[:, None] / np.maximum(row_scale[:, None], eps)
        + col_residual[None, :] / np.maximum(col_scale[None, :], eps)
    )
    residual_potential = np.exp(-0.22 * np.minimum(residual_cost, 8.0))
    residual_bonus = 0.90 + 0.16 * residual_potential

    # Neighborhood diversity. Low overlap suggests a bridge between
    # different local regions; its effect is deliberately weak and bounded.
    candidate_mask = np.zeros((n, n), dtype=float)
    candidate_mask[
        np.arange(n)[:, None],
        row_order[:, :k],
    ] = 1.0
    candidate_mask *= valid

    overlap = candidate_mask @ candidate_mask.T
    neighborhood_size = candidate_mask.sum(axis=1)
    union_size = (
        neighborhood_size[:, None]
        + neighborhood_size[None, :]
        - overlap
    )
    jaccard_overlap = overlap / np.maximum(union_size, 1.0)

    bridge_bonus = 1.0 + 0.10 * (1.0 - np.clip(jaccard_overlap, 0.0, 1.0))

    # Mutual nearest-neighbor edges remain a strong local cue.
    nearest_out = row_order[:, 0]
    nearest_in = col_order[0, :]
    mutual_neighbor = (
        nearest_out[:, None] == np.arange(n)[None, :]
    ) & (
        nearest_in[None, :] == np.arange(n)[:, None]
    )

    score = (
        base_score
        * endpoint_bonus
        * residual_bonus
        * bridge_bonus
    )
    score *= np.where(mutual_neighbor, 1.45, 1.0)
    score *= valid

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score = score / maximum

    np.fill_diagonal(score, 0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
