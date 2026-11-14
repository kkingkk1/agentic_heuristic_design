import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators based on two-hop shortcut indispensability."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    valid &= valid.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = d[valid]
    fallback = float(np.median(finite_values))
    if not np.isfinite(fallback) or fallback <= 0:
        positive = finite_values[finite_values > 0]
        fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, np.finfo(float).eps)

    symmetric_d = 0.5 * (d + d.T)
    safe_d = np.where(valid, symmetric_d, fallback * 1e6)
    np.fill_diagonal(safe_d, 0.0)

    positive = safe_d[valid & (safe_d > 0)]
    scale = float(np.median(positive)) if positive.size else fallback
    scale = max(scale, np.finfo(float).eps)

    two_hop = np.full((n, n), np.inf, dtype=float)
    alternative_count = np.zeros((n, n), dtype=float)

    for k in range(n):
        candidate = safe_d[:, [k]] + safe_d[[k], :]
        candidate[k, :] = np.inf
        candidate[:, k] = np.inf

        better = candidate < two_hop
        alternative_count += np.exp(
            -np.minimum(candidate / scale, 60.0)
        ) * np.isfinite(candidate)

        two_hop = np.minimum(two_hop, candidate)

    direct = np.where(valid, safe_d, np.inf)

    margin = two_hop - direct
    margin = np.where(np.isfinite(margin), margin, 0.0)

    temperature = max(0.25 * scale, np.finfo(float).eps)
    shortcut = 1.0 / (1.0 + np.exp(-np.clip(margin / temperature, -60.0, 60.0)))

    finite_two_hop = np.isfinite(two_hop)
    scarcity = 1.0 / (1.0 + alternative_count)
    scarcity *= finite_two_hop

    global_direct = np.exp(-np.minimum(direct / scale, 60.0))
    global_direct[~np.isfinite(global_direct)] = 0.0

    score = (
        0.60 * shortcut
        + 0.25 * scarcity
        + 0.15 * global_direct
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    for _ in range(8):
        row_sum = score.sum(axis=1, keepdims=True)
        score = np.divide(
            score,
            np.sqrt(row_sum * row_sum.T),
            out=np.zeros_like(score),
            where=(row_sum > 0) & (row_sum.T > 0),
        )
        score = 0.5 * (score + score.T)
        score[~valid] = 0.0
        np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
