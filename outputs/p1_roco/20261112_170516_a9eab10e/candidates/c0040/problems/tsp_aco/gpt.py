import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(raw) & (raw > 0)
    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    values = raw[finite_positive]
    fallback = float(np.median(values))
    fallback = max(fallback, np.finfo(float).eps)

    d = 0.5 * (raw + raw.T)
    valid = np.isfinite(d) & (d > 0)
    d = np.where(valid, d, fallback)
    np.fill_diagonal(d, np.inf)

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(finite))
    scale = max(scale, np.finfo(float).eps)

    # Best two-hop detour for every ordered pair, excluding the endpoints.
    two_hop = np.min(d[:, :, None] + d[None, :, :], axis=1)
    np.fill_diagonal(two_hop, np.inf)

    # A large positive value means the direct edge is a strong shortcut.
    shortcut = np.maximum(two_hop - d, 0.0)
    shortcut = shortcut / (scale + shortcut)

    # Compare the edge with the best alternative incident edge at each endpoint.
    alternatives_i = d.copy()
    alternatives_j = d.copy()
    np.fill_diagonal(alternatives_i, np.inf)
    np.fill_diagonal(alternatives_j, np.inf)

    best_i = np.min(alternatives_i, axis=1)
    best_j = np.min(alternatives_j, axis=1)

    # Low excess over endpoint alternatives indicates an inexpensive insertion.
    excess = d - 0.5 * (best_i[:, None] + best_j[None, :])
    insertion = np.exp(-np.maximum(excess, 0.0) / scale)

    # Mild absolute-length control, used only as a smooth temperature.
    length_term = np.exp(-d / (2.0 * scale))

    score = (0.60 + 0.40 * shortcut) * insertion * length_term
    score = 0.5 * (score + score.T)

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    if np.any(valid):
        score[~valid] = 0.0
        score[~valid.T] = 0.0

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
