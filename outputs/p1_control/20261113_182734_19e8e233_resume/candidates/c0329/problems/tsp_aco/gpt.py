import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank- and detour-aware priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    result = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    inf = np.inf
    masked = np.where(valid, d, inf)

    values = d[valid]
    positive = values[values > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, eps)

    order_out = np.argsort(masked, axis=1, kind="stable")
    rank_out = np.empty((n, n), dtype=float)
    positions = np.arange(1, n + 1, dtype=float)
    rank_out[np.arange(n)[:, None], order_out] = positions[None, :]

    order_in = np.argsort(masked, axis=0, kind="stable")
    rank_in = np.empty((n, n), dtype=float)
    rank_in[order_in, np.arange(n)[None, :]] = positions[:, None]

    rank_weight = (
        1.0 / (1.0 + rank_out)
        + 1.0 / (1.0 + rank_in)
    )
    rank_weight[~valid] = 0.0

    two_hop = np.full((n, n), inf, dtype=float)
    for k in range(n):
        candidate = masked[:, k, None] + masked[None, k, :]
        np.minimum(two_hop, candidate, out=two_hop)

    direct = np.where(valid, d, inf)
    detour_gap = two_hop - direct

    detour_scale = np.median(
        np.abs(detour_gap[np.isfinite(detour_gap) & valid])
    )
    if not np.isfinite(detour_scale) or detour_scale <= 0.0:
        detour_scale = scale
    detour_scale = max(float(detour_scale), eps)

    detour_advantage = np.zeros_like(d, dtype=float)
    np.divide(
        detour_gap,
        detour_scale,
        out=detour_advantage,
        where=valid & np.isfinite(detour_gap),
    )
    detour_advantage = np.clip(detour_advantage, -8.0, 8.0)
    detour_weight = 1.0 + 0.5 * np.tanh(detour_advantage)
    detour_weight[~valid] = 0.0

    outgoing_count = valid.sum(axis=1).astype(float)
    incoming_count = valid.sum(axis=0).astype(float)

    endpoint_weight = np.sqrt(
        (1.0 + outgoing_count[:, None])
        * (1.0 + incoming_count[None, :])
    )
    endpoint_weight /= max(float(endpoint_weight.max()), 1.0)
    endpoint_weight = 0.75 + 0.25 * endpoint_weight
    endpoint_weight[~valid] = 0.0

    score = rank_weight * detour_weight * endpoint_weight
    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(score.max())
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
