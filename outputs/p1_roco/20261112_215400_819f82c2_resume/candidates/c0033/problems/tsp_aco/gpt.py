import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structural shortcut-priority indicators for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite = np.isfinite(raw) & (raw >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite & off_diagonal

    values = raw[valid]
    if values.size == 0:
        return result

    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Build an undirected distance matrix from all valid directed observations.
    d = np.full((n, n), np.inf, dtype=float)
    d[valid] = raw[valid]
    d = np.minimum(d, d.T)
    np.fill_diagonal(d, np.inf)

    usable = np.isfinite(d)
    if not np.any(usable):
        return result

    # Global shortness signal.
    shortness = np.exp(-np.minimum(d, 50.0 * scale) / scale)

    # Cheapest two-hop detour for every edge.
    detour = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        left = d[:, k][:, None]
        right = d[k, :][None, :]
        candidate = left + right
        candidate[k, :] = np.inf
        candidate[:, k] = np.inf
        detour = np.minimum(detour, candidate)

    # Direct edges that strongly beat every two-hop alternative are structurally
    # difficult to replace and receive a larger prior.
    finite_detour = np.isfinite(detour)
    gap = np.zeros((n, n), dtype=float)
    gap[finite_detour] = (detour[finite_detour] - d[finite_detour]) / scale
    gap = np.clip(gap, -50.0, 50.0)
    necessity = 1.0 / (1.0 + np.exp(-gap))

    # If no third-node detour exists, retain the edge's shortness signal.
    necessity[~finite_detour & usable] = 1.0

    # Favor edges that are both short and structurally hard to replace.
    score = np.sqrt(shortness * (0.25 + 0.75 * necessity))
    score[~usable] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
