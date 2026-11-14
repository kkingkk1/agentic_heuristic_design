import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    positive = d[valid & (d > 0.0)]
    scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    scale = max(scale, np.finfo(float).eps)

    work = np.where(valid, d, np.inf)
    eps = np.finfo(float).eps

    # Robust global reference for local cycle costs.
    finite_values = work[np.isfinite(work)]
    typical = (
        float(np.median(finite_values))
        if finite_values.size
        else scale
    )
    typical = max(typical, eps)

    # For every edge i -> j, estimate the best valid continuation
    # j -> k and predecessor i <- h, excluding immediate reversal.
    successor = np.full((n, n), np.inf, dtype=float)
    predecessor = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        for j in range(n):
            if not valid[i, j]:
                continue

            outgoing = work[j].copy()
            outgoing[j] = np.inf
            outgoing[i] = np.inf
            successor[i, j] = np.min(outgoing)

            incoming = work[:, i].copy()
            incoming[i] = np.inf
            incoming[j] = np.inf
            predecessor[i, j] = np.min(incoming)

    continuation = successor + predecessor
    cycle_cost = d + continuation

    finite_cycle = np.isfinite(cycle_cost) & valid
    cycle_reference = (
        float(np.median(cycle_cost[finite_cycle]))
        if np.any(finite_cycle)
        else 3.0 * typical
    )
    cycle_reference = max(cycle_reference, eps)

    # A direct edge is attractive when two-hop replacements are substantially
    # more expensive. This measures shortcut consistency instead of raw rank.
    shortcut_margin = np.zeros((n, n), dtype=float)

    for i in range(n):
        intermediates = np.arange(n)
        intermediates = intermediates[(intermediates != i)]

        for j in range(n):
            if not valid[i, j]:
                continue

            candidates = intermediates[intermediates != j]
            if candidates.size == 0:
                continue

            left = work[i, candidates]
            right = work[candidates, j]
            replacement = left + right
            feasible = np.isfinite(replacement)

            if np.any(feasible):
                margins = replacement[feasible] - d[i, j]
                shortcut_margin[i, j] = float(np.median(margins))

    # Smoothly reward edges that dominate their two-hop replacements.
    shortcut_scale = max(scale, 0.5 * typical)
    shortcut_term = 1.0 / (
        1.0
        + np.exp(
            -np.clip(shortcut_margin / shortcut_scale, -30.0, 30.0)
        )
    )

    # Reward edges that participate in a cheap locally closable directed chain.
    normalized_cycle = (cycle_reference - cycle_cost) / scale
    cycle_term = 1.0 / (
        1.0
        + np.exp(-np.clip(normalized_cycle, -30.0, 30.0))
    )

    # Mild direct-distance factor prevents a structurally plausible but
    # extremely long edge from dominating.
    direct_term = np.exp(-np.clip(d / (2.0 * scale), 0.0, 30.0))

    score = shortcut_term * cycle_term * direct_term
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
