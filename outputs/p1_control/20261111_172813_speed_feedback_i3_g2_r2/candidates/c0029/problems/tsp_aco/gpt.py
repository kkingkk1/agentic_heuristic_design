import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute normalized edge-priority indicators for stochastic TSP sampling.
    Higher values indicate more promising edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(d)
    valid = finite & (d >= 0.0) & ~diagonal
    positive = valid & (d > 0.0)

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(d[positive]))
    epsilon = np.finfo(float).eps
    fallback = max(scale, epsilon) * 1e6

    safe_d = np.where(valid, np.maximum(d, epsilon), fallback)
    safe_d[diagonal] = np.inf

    inverse = np.zeros((n, n), dtype=float)
    finite_safe = np.isfinite(safe_d)
    inverse[finite_safe] = 1.0 / safe_d[finite_safe]

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max[row_max <= 0.0] = 1.0
    local = inverse / row_max

    order = np.argsort(-local, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(1, n + 1, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values

    rank_weight = 1.0 / np.log2(ranks + 1.0)
    rank_weight[diagonal] = 0.0

    reciprocal = 0.5 * (local + local.T)

    if n > 1:
        nearest = np.argmin(safe_d, axis=1)
        mutual = (
            (nearest[:, None] == np.arange(n)[None, :])
            & (nearest[None, :] == np.arange(n)[:, None])
            & valid
        ).astype(float)
    else:
        mutual = np.zeros((n, n), dtype=float)

    alternative = np.partition(safe_d, 1, axis=1)[:, 1]
    savings = np.maximum(
        alternative[:, None] + alternative[None, :] - safe_d,
        0.0,
    )
    savings[~valid] = 0.0

    savings_max = np.max(savings)
    if savings_max > 0.0:
        savings /= savings_max

    rank_signal = np.sqrt(rank_weight * rank_weight.T)
    local_signal = np.sqrt(np.maximum(local, 0.0) * np.maximum(local.T, 0.0))

    score = (
        0.35 * local_signal
        + 0.25 * reciprocal
        + 0.20 * savings
        + 0.10 * rank_signal
        + 0.10 * mutual
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
