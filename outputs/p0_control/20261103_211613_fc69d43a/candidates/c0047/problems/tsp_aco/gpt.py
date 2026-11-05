import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-preference scores for a minimization TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D NumPy array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    if np.any(distances[finite] < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    # Use an undirected distance representation and exclude self-loops.
    d = 0.5 * (distances + distances.T)
    valid = np.isfinite(d) & ~np.eye(n, dtype=bool)

    positive = d[valid & (d > 0.0)]
    if positive.size == 0:
        return scores

    global_scale = max(float(np.median(positive)), np.finfo(float).eps)

    # Replace unusable and zero-length edges for robust local statistics.
    safe = np.where(valid & (d > 0.0), d, np.nan)
    endpoint_scale = np.nanmedian(safe, axis=1)
    endpoint_scale = np.where(
        np.isfinite(endpoint_scale) & (endpoint_scale > 0.0),
        endpoint_scale,
        global_scale,
    )

    eps = np.finfo(float).eps
    normalized = d / global_scale
    local_ratio = d / np.sqrt(endpoint_scale[:, None] * endpoint_scale[None, :])

    affinity = np.zeros((n, n), dtype=float)
    usable = valid & (d > 0.0)
    affinity[usable] = np.exp(-local_ratio[usable]) / (normalized[usable] + eps)

    # Reward edges that are competitive at both endpoints.
    row_mass = affinity.sum(axis=1)
    mass_scale = float(np.median(row_mass[row_mass > 0.0])) if np.any(row_mass > 0.0) else 1.0
    scarcity = mass_scale / (row_mass + mass_scale)
    competitiveness = affinity / (np.max(affinity, axis=1, keepdims=True) + eps)
    endpoint_competition = np.sqrt(
        competitiveness * competitiveness.T
    )
    endpoint_scarcity = np.sqrt(scarcity[:, None] * scarcity[None, :])

    # Best replacement through a distinct third node.
    hop = d[:, :, None] + d[None, :, :]
    intermediary_valid = valid[:, :, None] & valid[None, :, :]
    hop = np.where(intermediary_valid, hop, np.inf)
    best_detour = np.min(hop, axis=1)

    edge_scale = normalized + eps
    regret = np.maximum(best_detour - normalized, 0.0) / edge_scale
    regret[~usable] = 0.0
    regret_scale = regret[regret > 0.0]
    regret_scale = float(np.median(regret_scale)) if regret_scale.size else 1.0
    regret = np.clip(regret / max(regret_scale, eps), 0.0, 1.0)

    raw = affinity * (
        0.55
        + 0.20 * endpoint_competition
        + 0.15 * endpoint_scarcity
        + 0.10 * regret
    )

    raw = 0.5 * (raw + raw.T)
    raw[~np.isfinite(raw)] = 0.0
    raw[~usable] = 0.0
    np.fill_diagonal(raw, 0.0)

    values = raw[raw > 0.0]
    if values.size:
        maximum = float(np.max(values))
        if maximum > 0.0:
            scores = raw / maximum

    np.fill_diagonal(scores, 0.0)
    return scores
