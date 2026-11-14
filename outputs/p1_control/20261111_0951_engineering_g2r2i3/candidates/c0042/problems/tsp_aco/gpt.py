import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = valid & (d > 0.0)
    row_scale = np.min(
        np.where(positive, d, np.inf),
        axis=1,
        keepdims=True,
    )
    row_scale = np.where(np.isfinite(row_scale), row_scale, 1.0)

    base = np.zeros_like(d, dtype=float)
    zero_edges = valid & (d == 0.0)
    base[zero_edges] = 1.0
    base[positive] = row_scale.repeat(n, axis=1)[positive] / d[positive]

    safe_d = np.where(valid, d, np.inf)
    order = np.argsort(safe_d, axis=1, kind="stable")

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_weight = np.zeros_like(d, dtype=float)
    rank_weight[valid] = 1.0 / (1.0 + ranks[valid])

    top_k = min(4, n - 1)
    topk = np.zeros((n, n), dtype=bool)
    topk[np.arange(n)[:, None], order[:, :top_k]] = True
    topk &= valid

    mutual_topk = topk & topk.T
    first_two = (ranks <= 1.0) & valid
    two_choice_support = first_two | first_two.T

    structural_factor = (
        0.84
        + 0.08 * rank_weight
        + 0.05 * mutual_topk.astype(float)
        + 0.03 * two_choice_support.astype(float)
    )

    score = base * structural_factor
    score += 0.02 * valid.astype(float)

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return score.astype(float, copy=False)
