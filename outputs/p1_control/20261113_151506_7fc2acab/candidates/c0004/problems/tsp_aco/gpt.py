import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Estimate how promising each directed TSP edge is."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    x = d.copy()
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=float)

    finite_values = x[finite]
    fallback = float(np.max(finite_values))
    if not np.isfinite(fallback):
        fallback = 1.0

    x[~finite] = fallback
    np.fill_diagonal(x, fallback)

    positive = x[x > 0]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    row_scale = np.median(x, axis=1)
    col_scale = np.median(x, axis=0)
    row_scale = np.maximum(row_scale, eps)
    col_scale = np.maximum(col_scale, eps)

    normalized = x / np.sqrt(row_scale[:, None] * col_scale[None, :])

    row_best = np.min(np.where(np.eye(n, dtype=bool), np.inf, x), axis=1)
    col_best = np.min(np.where(np.eye(n, dtype=bool), np.inf, x), axis=0)
    row_best = np.where(np.isfinite(row_best), row_best, global_scale)
    col_best = np.where(np.isfinite(col_best), col_best, global_scale)

    lookahead = x + row_best[None, :] + col_best[:, None]
    temperature = max(float(np.median(lookahead)), global_scale, eps)

    distance_score = np.exp(-np.minimum(normalized, 60.0))
    lookahead_score = np.exp(-np.minimum(lookahead / temperature, 60.0))

    row_rank = np.argsort(np.argsort(x, axis=1), axis=1)
    col_rank = np.argsort(np.argsort(x, axis=0), axis=0)
    rank_score = 1.0 / (
        1.0
        + row_rank.astype(float)
        + col_rank.astype(float)
    )

    nearest_out = np.argmin(
        np.where(np.eye(n, dtype=bool), np.inf, x), axis=1
    )
    nearest_in = np.argmin(
        np.where(np.eye(n, dtype=bool), np.inf, x), axis=0
    )

    mutual_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_out[i]
        mutual_bonus[i, j] += 1.0
        k = nearest_in[i]
        mutual_bonus[k, i] += 1.0

    score = (
        0.42 * distance_score
        + 0.28 * lookahead_score
        + 0.20 * rank_score
        + 0.10 * mutual_bonus
    )

    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
