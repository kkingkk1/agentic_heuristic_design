import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros_like(d)

    scale = np.median(finite)
    if not np.isfinite(scale) or scale <= 0:
        positive = finite[finite > 0]
        scale = np.median(positive) if positive.size else 1.0

    safe_d = np.where(np.isfinite(d), d, 10.0 * scale)
    positive = safe_d[safe_d > 0]
    base = np.median(positive) if positive.size else scale
    base = max(base, np.finfo(float).eps)

    affinity = (
        np.exp(-safe_d / (0.35 * base))
        + np.exp(-safe_d / (0.85 * base))
        + 0.5 * np.exp(-safe_d / (1.80 * base))
    )

    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)

    mutual = np.sqrt(rank_score * rank_score.T)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reciprocal = 1.0 / np.maximum(safe_d, np.finfo(float).eps)
    reciprocal[~np.isfinite(reciprocal)] = 0.0

    two_step = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidate_cost = safe_d[i, :, None] + safe_d[None, :, :]
        candidate_cost[:, i] = np.inf
        candidate_cost[i, :] = np.inf
        two_step[i] = np.min(candidate_cost, axis=0)

    two_step = np.where(np.isfinite(two_step), two_step, 10.0 * base)
    lookahead = 1.0 / np.maximum(two_step, np.finfo(float).eps)

    score = (
        0.55 * affinity
        + 0.25 * mutual
        + 0.12 * reciprocal / (np.max(reciprocal) + np.finfo(float).eps)
        + 0.08 * lookahead / (np.max(lookahead) + np.finfo(float).eps)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float)
