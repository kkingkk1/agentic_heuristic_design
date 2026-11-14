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
    finite_nonnegative = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(finite_nonnegative):
        return np.zeros((n, n), dtype=float)

    safe = np.where(finite_nonnegative, distances, np.inf)

    # A usable undirected edge must be valid in at least one direction.
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    # Stable ranks over finite neighbors only.
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

    valid_counts = np.sum(valid, axis=1)

    def neighborhood_scale(width: int) -> np.ndarray:
        width = min(width, n - 1)
        if width <= 0:
            return np.full(n, global_scale, dtype=float)

        nearest = np.take_along_axis(safe, row_order[:, :width], axis=1)
        finite = np.isfinite(nearest)
        values = np.where(finite, nearest, 0.0)
        counts = np.sum(finite, axis=1)

        scale = np.sum(values, axis=1) / np.maximum(counts, 1)
        scale = np.where(counts > 0, scale, global_scale)
        return np.maximum(scale, eps)

    local_scale = neighborhood_scale(min(4, n - 1))
    broad_scale = neighborhood_scale(min(8, n - 1))
    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    normalized_distance = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(-np.clip(normalized_distance, 0.0, 50.0))

    # Endpoint-wise distance affinity preserves local scale differences.
    endpoint_affinity = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 50.0))
    )

    # Bidirectional rank quality is a restrained alternative to stacking many
    # highly correlated local bonuses.
    reciprocal_rank = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + col_rank)
    )
    rank_signal = 0.55 + 0.45 * reciprocal_rank

    nearest_row = row_order[:, 0]
    nearest_col = col_order[0, :]

    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_col[None, :] == np.arange(n)[:, None])
    )

    k_local = min(2, n - 1)
    k_broad = min(4, n - 1)

    local_neighbors = valid & (row_rank < float(k_local))
    broad_neighbors = valid & (row_rank < float(k_broad))

    mutual_local = local_neighbors & local_neighbors.T
    mutual_broad = broad_neighbors & broad_neighbors.T

    # Multi-scale support is weak and only applies to edges that are already
    # competitive from both endpoints.
    bidirectional_competitive = (
        valid
        & (row_rank < float(k_broad))
        & (col_rank < float(k_broad))
    )

    shared_local = (
        local_neighbors.astype(float)
        @ local_neighbors.astype(float).T
    ) / max(float(k_local), 1.0)

    shared_broad = (
        broad_neighbors.astype(float)
        @ broad_neighbors.astype(float).T
    ) / max(float(k_broad), 1.0)

    mutual_bonus = (
        1.0
        + 0.22 * mutual_nearest.astype(float)
        + 0.08 * mutual_local.astype(float)
        + 0.05 * mutual_broad.astype(float)
    )

    validated_multiscale = (
        1.0
        + 0.045
        * np.clip(0.65 * shared_local + 0.35 * shared_broad, 0.0, 1.0)
        * bidirectional_competitive.astype(float)
    )

    # Opportunity cost is based on the best endpoint neighborhood and is kept
    # mild so that distance remains the dominant factor.
    endpoint_best = np.minimum(local_scale[:, None], local_scale[None, :])
    opportunity_ratio = safe / np.maximum(endpoint_best, eps)
    opportunity_penalty = 1.0 / (
        1.0 + 0.12 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    # Weak bridge-like support: an edge is useful when both endpoints have
    # somewhat different local scales, but the effect is capped.
    scale_mismatch = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    bridge_signal = np.clip(
        scale_mismatch / (1.0 + scale_mismatch),
        0.0,
        1.0,
    )
    bridge_adjustment = (
        1.0
        + 0.035
        * bridge_signal
        * bidirectional_competitive.astype(float)
    )

    # Preserve alternatives around concentrated or sparse neighborhoods.
    local_strength = np.where(
        valid,
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        / np.sqrt(1.0 + row_rank),
        0.0,
    )

    strongest = np.max(local_strength, axis=1)
    total = np.sum(local_strength, axis=1)
    concentration = strongest / np.maximum(total, eps)

    sorted_strength = np.sort(local_strength, axis=1)
    second = sorted_strength[:, -2] if n >= 2 else np.zeros(n)
    alternative_ratio = np.clip(
        second / np.maximum(strongest, eps),
        0.0,
        1.0,
    )

    sparsity = np.clip(
        local_scale / np.maximum(global_scale, eps),
        0.0,
        2.0,
    )
    sparsity = sparsity / (1.0 + sparsity)

    edge_topness = np.sqrt(
        np.clip(1.0 / (1.0 + row_rank), 0.0, 1.0)
        * np.clip(1.0 / (1.0 + col_rank), 0.0, 1.0)
    )

    alternative_support = np.sqrt(
        alternative_ratio[:, None] * alternative_ratio[None, :]
    )
    sparse_support = np.sqrt(
        sparsity[:, None] * sparsity[None, :]
    )

    # Penalize only excessive concentration and only on already top-ranked
    # edges, leaving strong concentrated structures mostly intact.
    concentration_penalty = (
        np.sqrt(concentration[:, None] * concentration[None, :])
        * edge_topness
    )

    base_scores = (
        distance_score
        * endpoint_affinity
        * rank_signal
        * mutual_bonus
        * validated_multiscale
        * opportunity_penalty
        * bridge_adjustment
    )

    base_scores[~valid] = 0.0
    base_scores = np.maximum(base_scores, eps)

    log_scores = np.log(base_scores)
    log_scores += 0.045 * alternative_support * bidirectional_competitive
    log_scores += 0.025 * sparse_support * (1.0 - edge_topness)
    log_scores -= 0.055 * concentration_penalty

    scores = np.exp(np.clip(log_scores, -60.0, 30.0))
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
