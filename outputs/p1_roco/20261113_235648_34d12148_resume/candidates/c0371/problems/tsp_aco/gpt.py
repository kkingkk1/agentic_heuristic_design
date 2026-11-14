import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return compact distance- and rank-based priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    valid = offdiag & np.isfinite(raw)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = np.maximum(raw[valid], 0.0)

    observed = distances[valid]
    positive = observed[observed > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    upper = float(np.quantile(observed, 0.95))
    if not np.isfinite(upper) or upper <= 0.0:
        upper = scale

    clipped = np.minimum(distances, upper)
    temperature = max(0.45 * scale, np.finfo(float).eps)

    affinity = np.exp(-clipped / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    inf = np.inf
    ranked_costs = np.where(valid, clipped, inf)

    row_order = np.argsort(ranked_costs, axis=1, kind="mergesort")
    row_rank = np.empty_like(ranked_costs, dtype=float)
    row_rank[row_order] = np.arange(n, dtype=float)[None, :]
    row_rank[~valid] = inf

    column_order = np.argsort(ranked_costs, axis=0, kind="mergesort")
    column_rank = np.empty_like(ranked_costs, dtype=float)
    column_rank[column_order, np.arange(n)] = np.arange(n, dtype=float)[:, None]
    column_rank[~valid] = inf

    row_counts = np.maximum(valid.sum(axis=1).astype(float), 1.0)
    column_counts = np.maximum(valid.sum(axis=0).astype(float), 1.0)

    row_tau = np.maximum(2.0, np.sqrt(row_counts))
    column_tau = np.maximum(2.0, np.sqrt(column_counts))

    row_signal = np.exp(-row_rank / row_tau[:, None])
    column_signal = np.exp(-column_rank / column_tau[None, :])

    row_signal[~valid] = 0.0
    column_signal[~valid] = 0.0

    rank_signal = np.sqrt(row_signal * column_signal)

    score = (
        0.74 * affinity
        + 0.23 * rank_signal
        + 0.03 * valid.astype(float)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
