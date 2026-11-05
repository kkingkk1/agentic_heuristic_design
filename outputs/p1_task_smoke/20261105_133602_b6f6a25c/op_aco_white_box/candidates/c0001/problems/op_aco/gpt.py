import numpy as np

def heuristics_v2(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    """
    Generate a prize- and budget-aware edge heuristic for the Orienteering Problem.

    The returned matrix H satisfies H[i, j] >= 0 and H[i, i] = 0.
    Larger values indicate more promising transitions from node i to node j.
    """
    prize = np.asarray(prize, dtype=float)
    distance = np.asarray(distance, dtype=float)

    n = prize.shape[0]
    if distance.shape != (n, n):
        raise ValueError("distance must have shape (n, n)")
    if n == 0:
        return np.empty((0, 0), dtype=float)

    budget = float(maxlen)
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("maxlen must be a positive finite scalar")

    d = np.maximum(distance, 0.0)
    finite_d = d[np.isfinite(d)]
    positive_d = finite_d[finite_d > 0]

    if positive_d.size:
        distance_scale = float(np.median(positive_d))
    else:
        distance_scale = 1.0

    budget_scale = max(budget, distance_scale, np.finfo(float).eps)

    p = np.maximum(np.nan_to_num(prize, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    prize_scale = float(np.max(p))
    if prize_scale > 0:
        p_norm = p / prize_scale
    else:
        p_norm = np.zeros(n, dtype=float)

    valid = np.isfinite(d) & (d > 0)
    d_safe = np.where(valid, d, np.inf)

    shortness = np.exp(-np.minimum(d_safe, 50.0 * budget_scale) / budget_scale)
    shortness[~np.isfinite(shortness)] = 0.0

    budget_feasibility = np.clip(1.0 - d_safe / budget_scale, 0.0, 1.0)
    budget_feasibility[~np.isfinite(budget_feasibility)] = 0.0

    local_opportunity = np.zeros(n, dtype=float)
    for i in range(n):
        reachable = valid[i] & (d[i] <= budget)
        if np.any(reachable):
            weights = np.exp(-d[i, reachable] / budget_scale)
            local_opportunity[i] = float(
                np.sum(weights * p_norm[reachable])
                / max(np.sum(weights), np.finfo(float).eps)
            )

    destination_value = p_norm[None, :]
    source_value = p_norm[:, None]
    neighborhood_value = 0.5 * (
        local_opportunity[:, None] + local_opportunity[None, :]
    )

    edge_value = (
        0.50 * destination_value
        + 0.20 * source_value
        + 0.20 * neighborhood_value
        + 0.10 * np.maximum(destination_value, source_value)
    )

    h = edge_value * (0.65 * shortness + 0.35 * budget_feasibility)

    if n > 1:
        depot_edges = np.zeros((n, n), dtype=bool)
        depot_edges[0, 1:] = True
        depot_edges[1:, 0] = True
        depot_value = np.maximum(p_norm[:, None], p_norm[None, :])
        h = np.where(depot_edges, 0.75 * depot_value * (shortness + budget_feasibility), h)

    h[~valid] = 0.0
    np.fill_diagonal(h, 0.0)

    finite_h = h[np.isfinite(h)]
    if finite_h.size:
        scale = float(np.max(finite_h))
        if scale > 0:
            h = h / scale

    return np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
