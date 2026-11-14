import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge priorities using global efficiency and detour utility."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)
    valid = finite & finite.T

    if not np.any(valid):
        return scores

    with np.errstate(all="ignore"):
        d = 0.5 * (distances + distances.T)

    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    values = d[valid]
    positive = values[values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        scale = float(np.median(positive))
    elif values.size:
        scale = float(np.max(values))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scale = max(scale, eps)

    finite_max = float(np.max(values))
    fallback = max(finite_max, scale) * 4.0 + scale
    work = np.where(valid, d, fallback)
    np.fill_diagonal(work, fallback)

    direct_efficiency = np.exp(
        -np.clip(work / scale, 0.0, 60.0)
    )

    two_hop = np.min(
        work[:, :, None] + work[None, :, :],
        axis=1,
    )

    replacement = np.maximum(two_hop, eps)
    detour_ratio = np.divide(
        replacement - work,
        replacement,
        out=np.zeros((n, n), dtype=float),
        where=valid & (replacement > 0.0),
    )
    detour_ratio = np.clip(detour_ratio, 0.0, 1.0)

    indispensability = 0.72 + 0.58 * detour_ratio

    degree = valid.sum(axis=1).astype(float)
    degree_scale = max(float(np.mean(degree[degree > 0.0])) if np.any(degree > 0.0) else 1.0, 1.0)

    endpoint_load = np.sqrt(
        np.maximum(degree[:, None], 1.0)
        * np.maximum(degree[None, :], 1.0)
    )
    balance = np.divide(
        degree_scale,
        endpoint_load,
        out=np.ones((n, n), dtype=float),
        where=endpoint_load > 0.0,
    )
    balance = np.clip(balance, 0.72, 1.18)

    local_mean = np.divide(
        np.where(valid, d, 0.0).sum(axis=1),
        np.maximum(degree, 1.0),
    )
    contrast = np.sqrt(
        np.maximum(local_mean[:, None], eps)
        * np.maximum(local_mean[None, :], eps)
    )
    relative_global_cost = np.divide(
        d,
        np.maximum(contrast, eps),
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )
    cost_gate = 1.0 / (
        1.0 + np.maximum(relative_global_cost - 1.0, 0.0)
    )
    cost_gate = 0.75 + 0.25 * np.clip(cost_gate, 0.0, 1.0)

    scores = (
        direct_efficiency
        * indispensability
        * balance
        * cost_gate
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
