import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n < 2:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    finite_values = d[valid]
    global_scale = float(np.median(finite_values))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    safe_d = np.where(valid, d, np.inf)

    row_scales = np.full(n, global_scale, dtype=float)
    for i in range(n):
        values = safe_d[i, valid[i]]
        if values.size:
            scale = np.median(values)
            if np.isfinite(scale) and scale > 0:
                row_scales[i] = scale

    logits = -safe_d / row_scales[:, None]
    logits[~valid] = -np.inf

    row_max = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.where(valid, logits - row_max, -np.inf))
    row_prob = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    col_prob = row_prob.T
    scarcity = np.sqrt(row_prob * col_prob)
    scarcity[~valid] = 0.0

    two_hop = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        candidate = safe_d[:, k, None] + safe_d[None, k, :]
        two_hop = np.minimum(two_hop, candidate)

    direct = np.where(valid, d, np.inf)
    improvement = (two_hop - direct) / (
        global_scale + 0.5 * np.minimum(row_scales[:, None], row_scales[None, :])
    )
    bridge = 1.0 / (1.0 + np.exp(-np.clip(improvement, -60.0, 60.0)))
    bridge[~valid] = 0.0

    score = 0.65 * scarcity + 0.35 * bridge
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = score.max()
    if maximum > 0:
        score /= maximum

    return score
