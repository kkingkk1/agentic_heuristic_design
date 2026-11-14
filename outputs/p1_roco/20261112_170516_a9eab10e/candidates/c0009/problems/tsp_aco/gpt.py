import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)

    safe_d = np.where(positive, d, np.nan)
    fallback = np.nanmedian(safe_d)
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    safe_d = np.where(positive, d, fallback)
    np.fill_diagonal(safe_d, np.inf)

    # Robust distance scale prevents a few outliers from dominating the scores.
    finite_offdiag = safe_d[np.isfinite(safe_d)]
    scale = np.percentile(finite_offdiag, 75) if finite_offdiag.size else fallback
    scale = max(float(scale), np.finfo(float).eps)

    # Smooth inverse-distance attraction.
    attraction = 1.0 / (1.0 + safe_d / scale)
    attraction[~np.isfinite(attraction)] = 0.0

    # Local rank score: edges near the top of each node's neighborhood rank higher.
    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n, dtype=float)[None, :]
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(n)))
    rank_score[~np.isfinite(rank_score)] = 0.0

    # Mutual-neighborhood agreement rewards edges attractive from both endpoints.
    mutual = np.sqrt(rank_score * rank_score.T)

    # Mild hub penalty discourages repeatedly selecting edges incident to one node.
    row_mean = np.mean(attraction, axis=1, keepdims=True)
    row_mean = np.maximum(row_mean, np.finfo(float).eps)
    relative = attraction / row_mean
    hub_penalty = 1.0 / np.sqrt(
        1.0 + np.maximum(relative.sum(axis=1, keepdims=True) - 1.0, 0.0)
    )
    hub_factor = np.sqrt(hub_penalty * hub_penalty.T)

    score = (
        0.50 * attraction
        + 0.35 * mutual
        + 0.15 * np.sqrt(np.maximum(attraction * mutual, 0.0))
    )
    score *= hub_factor
    score = 0.5 * (score + score.T)

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
