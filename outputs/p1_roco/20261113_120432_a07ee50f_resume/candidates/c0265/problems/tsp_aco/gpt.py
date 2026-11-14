import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    raw_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(raw_valid):
        return result

    raw = np.where(raw_valid, distances, np.inf)

    # Convert directed input into a symmetric undirected distance graph.
    safe = np.minimum(raw, raw.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return result

    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite = safe[valid]
        global_scale = float(np.median(finite)) if finite.size else 1.0

    eps = np.finfo(float).eps
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    # Global short-edge preference; unlike local-neighborhood methods,
    # every edge is measured against one shared scale.
    normalized_cost = np.clip(safe / global_scale, 0.0, 60.0)
    direct_preference = np.exp(-normalized_cost)

    # Find the best two-hop detour for every pair of endpoints.
    # The intermediate node is required to differ from both endpoints.
    best_detour = np.full((n, n), np.inf, dtype=float)

    for middle in range(n):
        left = safe[:, middle][:, None]
        right = safe[middle, :][None, :]
        candidate = left + right

        allowed = np.ones((n, n), dtype=bool)
        allowed[middle, :] = False
        allowed[:, middle] = False

        candidate[~allowed] = np.inf
        best_detour = np.minimum(best_detour, candidate)

    np.fill_diagonal(best_detour, np.inf)

    finite_detour = np.isfinite(best_detour) & valid

    # Positive excess means that the direct edge beats its best two-hop
    # replacement. A bounded sigmoid prevents this structural signal from
    # overwhelming the actual distance.
    detour_margin = np.zeros((n, n), dtype=float)
    detour_margin[finite_detour] = (
        best_detour[finite_detour] - safe[finite_detour]
    ) / global_scale
    detour_margin = np.clip(detour_margin, -12.0, 12.0)

    necessity = 1.0 / (1.0 + np.exp(-detour_margin))

    # Edges without a two-hop alternative receive a restrained bridge bonus.
    bridge_bonus = np.where(
        valid & ~finite_detour,
        1.12,
        1.0,
    )

    # Compute a global affinity load for each endpoint. This mildly reduces
    # edges attached to endpoints that are broadly compatible with many
    # other nodes, preserving diversity without using neighborhood ranks.
    affinity = np.where(valid, direct_preference, 0.0)
    endpoint_load = np.sum(affinity, axis=1)
    mean_load = max(float(np.mean(endpoint_load[endpoint_load > 0])), eps)

    load_ratio = np.clip(endpoint_load / mean_load, 0.0, 6.0)
    load_factor = 1.0 / np.sqrt(
        (1.0 + 0.07 * load_ratio[:, None])
        * (1.0 + 0.07 * load_ratio[None, :])
    )

    # Keep the structural correction bounded and distance-dominated.
    scores = (
        direct_preference
        * (0.72 + 0.28 * necessity)
        * bridge_bonus
        * load_factor
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
