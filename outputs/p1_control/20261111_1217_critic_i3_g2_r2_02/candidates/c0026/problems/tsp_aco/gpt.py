import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-inclusion priorities for stochastic TSP solution sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_off_diagonal = np.isfinite(distances)
    np.fill_diagonal(finite_off_diagonal, False)

    if not np.any(finite_off_diagonal):
        return np.zeros((n, n), dtype=float)

    finite_values = distances[finite_off_diagonal]
    fallback = float(np.max(finite_values)) if finite_values.size else 1.0
    fallback = max(fallback, 0.0)

    d = distances.copy()
    d[~np.isfinite(d)] = fallback
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive_values = d[np.isfinite(d) & (d > 0.0)]
    global_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    global_scale = max(global_scale, np.finfo(float).eps)

    # Weak global attraction keeps non-candidate finite edges available.
    global_attraction = 1.0 / (1.0 + d / global_scale)
    global_attraction[~np.isfinite(global_attraction)] = 0.0

    # Select a sparse local candidate set without sorting every row.
    candidate_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    candidate_indices = np.argpartition(
        d,
        kth=candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    row_indices = np.arange(n)[:, None]
    candidate_distances = d[row_indices, candidate_indices]

    # Sort only the small candidate blocks to obtain local ranks.
    local_order = np.argsort(candidate_distances, axis=1, kind="stable")
    candidate_indices = np.take_along_axis(
        candidate_indices,
        local_order,
        axis=1,
    )
    candidate_distances = np.take_along_axis(
        candidate_distances,
        local_order,
        axis=1,
    )

    local_scales = np.median(candidate_distances, axis=1)
    local_scales = np.where(
        np.isfinite(local_scales) & (local_scales > 0.0),
        local_scales,
        global_scale,
    )
    local_scales = np.maximum(local_scales, np.finfo(float).eps)

    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_mask[row_indices, candidate_indices] = True
    np.fill_diagonal(candidate_mask, False)

    mutual = candidate_mask & candidate_mask.T
    one_sided = candidate_mask ^ candidate_mask.T

    candidate_score = np.zeros((n, n), dtype=float)

    for rank in range(candidate_count):
        sources = np.arange(n)
        targets = candidate_indices[:, rank]
        edge_distances = d[sources, targets]

        pair_scale = np.sqrt(
            local_scales[sources] * local_scales[targets]
        )
        pair_scale = np.maximum(pair_scale, np.finfo(float).eps)

        local_attraction = 1.0 / (1.0 + edge_distances / pair_scale)
        rank_score = 1.0 / (1.0 + float(rank))

        # Nodes with larger local scales are locally isolated; mildly
        # favor their useful outgoing connections to preserve bridges.
        source_sparsity = np.sqrt(
            local_scales[sources] / global_scale
        )
        target_sparsity = np.sqrt(
            local_scales[targets] / global_scale
        )
        bridge_score = np.sqrt(source_sparsity * target_sparsity)
        bridge_score = np.clip(bridge_score, 0.5, 2.0) / 2.0

        value = (
            0.58 * local_attraction
            + 0.24 * rank_score
            + 0.18 * bridge_score
        )
        candidate_score[sources, targets] = np.maximum(
            candidate_score[sources, targets],
            value,
        )

    mutual_bonus = mutual.astype(float)
    one_sided_bonus = one_sided.astype(float)

    score = (
        0.22 * global_attraction
        + 0.58 * candidate_score
        + 0.15 * mutual_bonus
        + 0.05 * one_sided_bonus
    )

    score = np.maximum(score, 0.0)
    score = (score + score.T) / 2.0
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
