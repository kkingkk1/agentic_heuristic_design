import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)

    # An undirected edge is usable only when both directions are valid.
    valid = finite_nonnegative & finite_nonnegative.T & off_diagonal

    if not np.any(valid):
        return result

    # TSP is treated as an undirected problem. Invalid entries are never used.
    symmetric_distance = 0.5 * (
        distance_matrix + distance_matrix.T
    )

    valid_values = symmetric_distance[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
        if not np.isfinite(global_scale) or global_scale <= 0.0:
            global_scale = float(np.mean(positive_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps

    # Robust global affinity. Zero-distance valid edges receive affinity 1.
    global_affinity = np.zeros((n, n), dtype=float)
    global_affinity[valid] = np.exp(
        -symmetric_distance[valid] / (global_scale + eps)
    )

    # Row-local scales and ranks are computed from valid edges only.
    local_scale = np.full(n, global_scale, dtype=float)
    rank_affinity = np.zeros((n, n), dtype=float)
    row_affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_mask = valid[i]
        row_values = symmetric_distance[i, row_mask]

        if row_values.size:
            ordered = np.sort(row_values)

            # A small nearest-neighbor set preserves local contrast while
            # remaining stable against one anomalous edge.
            local_count = min(3, ordered.size)
            local_candidates = ordered[:local_count]
            positive_local = local_candidates[local_candidates > 0.0]

            if positive_local.size:
                scale = float(np.median(positive_local))
            else:
                scale = global_scale

            if not np.isfinite(scale) or scale <= 0.0:
                scale = global_scale

            local_scale[i] = scale

            valid_indices = np.flatnonzero(row_mask)
            row_order = valid_indices[np.argsort(symmetric_distance[i, valid_indices])]
            for rank, j in enumerate(row_order):
                rank_affinity[i, j] = 1.0 / (1.0 + float(rank))

            row_affinity[i, row_mask] = np.exp(
                -symmetric_distance[i, row_mask] / (scale + eps)
            )

    # Combine endpoint-local evidence symmetrically.
    local_affinity = np.sqrt(
        np.maximum(row_affinity, 0.0)
        * np.maximum(row_affinity.T, 0.0)
    )

    symmetric_rank = np.sqrt(
        np.maximum(rank_affinity, 0.0)
        * np.maximum(rank_affinity.T, 0.0)
    )

    local_neighbor_affinity = 0.7 * local_affinity + 0.3 * symmetric_rank

    # Edge-specific savings: compare the candidate with the best alternative
    # edge available at each endpoint, excluding the candidate itself.
    savings = np.zeros((n, n), dtype=float)

    for i in range(n):
        valid_indices = np.flatnonzero(valid[i])
        for j in valid_indices:
            alternatives_i = valid_indices[valid_indices != j]
            if alternatives_i.size == 0:
                continue

            alternatives_j = np.flatnonzero(valid[:, j])
            alternatives_j = alternatives_j[alternatives_j != i]
            if alternatives_j.size == 0:
                continue

            alternative_i = float(
                np.min(symmetric_distance[i, alternatives_i])
            )
            alternative_j = float(
                np.min(symmetric_distance[alternatives_j, j])
            )
            candidate = float(symmetric_distance[i, j])

            denominator = alternative_i + alternative_j + eps
            edge_saving = (
                alternative_i + alternative_j - 2.0 * candidate
            ) / denominator

            savings[i, j] = max(edge_saving, 0.0)

    savings = 0.5 * (savings + savings.T)

    positive_savings = savings[valid & (savings > 0.0)]
    if positive_savings.size:
        savings_scale = float(np.percentile(positive_savings, 90))
        if not np.isfinite(savings_scale) or savings_scale <= 0.0:
            savings_scale = float(np.max(positive_savings))
        normalized_savings = np.clip(
            savings / (savings_scale + eps), 0.0, 1.0
        )
    else:
        normalized_savings = np.zeros_like(savings)

    score = (
        0.40 * global_affinity
        + 0.40 * local_neighbor_affinity
        + 0.20 * normalized_savings
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    return score
