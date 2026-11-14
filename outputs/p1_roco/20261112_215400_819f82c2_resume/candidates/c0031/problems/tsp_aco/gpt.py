import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = d[valid]
    positive = values[values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = float(np.max(values)) if values.size else 1.0

    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, np.inf)

    # Entropic edge affinity.
    temperature = scale
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-np.clip(safe[valid] / temperature, 0.0, 60.0))

    # Symmetric matrix scaling produces globally balanced edge support.
    row_mass = affinity.sum(axis=1)
    active = row_mass > 0.0

    if not np.any(active):
        return np.zeros((n, n), dtype=float)

    scaling = np.ones(n, dtype=float)

    for _ in range(120):
        weighted_mass = affinity @ scaling
        update = np.ones(n, dtype=float)
        usable = weighted_mass > 0.0
        update[usable] = 1.0 / np.sqrt(weighted_mass[usable])
        update[~active] = 0.0

        if np.max(np.abs(update - 1.0)) < 1e-10:
            break

        scaling *= update
        scaling = np.clip(scaling, 0.0, 1e12)

    balanced = affinity * scaling[:, None] * scaling[None, :]
    balanced = 0.5 * (balanced + balanced.T)

    # Direct-edge advantage over the best valid two-hop detour.
    two_hop = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        left = safe[:, k][:, None]
        right = safe[k, :][None, :]
        candidate = left + right
        two_hop = np.minimum(two_hop, candidate)

    detour_margin = np.zeros((n, n), dtype=float)
    comparable = valid & np.isfinite(two_hop)
    detour_margin[comparable] = np.maximum(
        two_hop[comparable] - safe[comparable],
        0.0,
    )

    direct_advantage = np.ones((n, n), dtype=float)
    direct_advantage[comparable] = 1.0 - np.exp(
        -detour_margin[comparable] / scale
    )

    # Retain a positive baseline so valid edges remain sampleable.
    score = balanced * (0.35 + 0.65 * direct_advantage)
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
