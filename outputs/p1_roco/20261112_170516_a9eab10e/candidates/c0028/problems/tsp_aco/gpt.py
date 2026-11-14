import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    np.fill_diagonal(d, np.nan)

    finite_positive = d[np.isfinite(d) & (d > 0)]
    if finite_positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_positive) * 10.0)
    d[~np.isfinite(d)] = fallback
    d[d < 0] = fallback

    positive = d[(d > 0) & np.isfinite(d)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    order = np.argsort(d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    row_ids = np.arange(n)[:, None]
    ranks[row_ids, order] = np.arange(n, dtype=float)[None, :]
    ranks[np.arange(n), np.arange(n)] = n

    rank_score = np.exp(-0.7 * (ranks - 1.0) / max(n - 2, 1))
    rank_score[np.arange(n), np.arange(n)] = 0.0

    nearest = np.min(np.where(np.eye(n, dtype=bool), np.inf, d), axis=1)
    second = np.partition(
        np.where(np.eye(n, dtype=bool), np.inf, d), 1, axis=1
    )[:, 1]
    second = np.maximum(second, nearest + np.finfo(float).eps)

    scarcity = np.clip((second[:, None] - d) / second[:, None], -1.0, 1.0)
    scarcity = 1.0 / (1.0 + np.exp(-4.0 * scarcity))
    scarcity[np.arange(n), np.arange(n)] = 0.0

    temperatures = np.array([0.45, 0.9, 1.8]) * scale
    distance_score = np.zeros((n, n), dtype=float)

    for temperature in temperatures:
        distance_score += np.exp(-d / max(temperature, np.finfo(float).eps))

    distance_score /= len(temperatures)
    distance_score[np.arange(n), np.arange(n)] = 0.0

    mutuality = np.sqrt(rank_score * rank_score.T)
    endpoint_scarcity = np.sqrt(scarcity * scarcity.T)

    score = (
        0.50 * distance_score
        + 0.30 * mutuality
        + 0.20 * endpoint_scarcity
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score.astype(float)
