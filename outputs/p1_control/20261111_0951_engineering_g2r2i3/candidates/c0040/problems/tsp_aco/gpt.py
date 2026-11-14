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

    positive = d[valid & (d > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps

    safe_d = np.where(valid, np.maximum(d, 0.0), np.inf)

    order = np.argsort(safe_d, axis=1, kind="stable")
    sorted_d = np.take_along_axis(safe_d, order, axis=1)

    scale_k = min(4, n - 1)
    local = sorted_d[:, :scale_k]
    local = np.where(np.isfinite(local), local, fallback)
    scale = np.maximum(np.mean(local, axis=1), eps)

    pair_scale = np.sqrt(scale[:, None] * scale[None, :])
    normalized = np.divide(
        safe_d,
        np.maximum(pair_scale, eps),
        out=np.full_like(safe_d, np.inf, dtype=float),
        where=np.isfinite(safe_d),
    )

    affinity = np.exp(-normalized)
    broad_affinity = np.exp(-0.35 * normalized)

    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_weight = 1.0 / (1.0 + ranks)
    rank_weight[~valid] = 0.0
    rank_weight /= np.maximum(
        np.max(rank_weight, axis=1, keepdims=True),
        eps,
    )

    reciprocal_rank = np.sqrt(rank_weight * rank_weight.T)

    top_k = min(4, n - 1)
    topk = np.zeros((n, n), dtype=bool)
    topk[np.arange(n)[:, None], order[:, :top_k]] = True
    topk &= valid

    mutual_topk = topk & topk.T
    mutual_top1 = mutual_topk & (ranks == 0.0) & (ranks.T == 0.0)

    first_two = (ranks <= 1.0) & valid
    two_choice_support = np.maximum(first_two, first_two.T).astype(float)

    mutual_bonus = (
        1.0
        + 0.28 * mutual_topk.astype(float)
        + 0.16 * mutual_top1.astype(float)
    )

    rank_factor = 0.45 + 0.55 * reciprocal_rank
    diversity_factor = 0.88 + 0.12 * two_choice_support
    distance_factor = 0.55 + 0.45 * broad_affinity

    score = (
        affinity
        * distance_factor
        * rank_factor
        * mutual_bonus
        * diversity_factor
    )

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
