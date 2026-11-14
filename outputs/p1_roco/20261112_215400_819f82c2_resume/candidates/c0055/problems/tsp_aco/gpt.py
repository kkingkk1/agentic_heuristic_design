import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-priority indicators for stochastic TSP solution sampling.
    Higher values indicate more promising directed edges.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    finite = d[valid]
    positive = finite[finite > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    elif finite.size:
        scale = max(float(np.median(finite)), 1.0)
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)
    safe = np.where(valid, d, np.inf)

    # Robust global distance desirability.
    distance_score = np.exp(-np.clip(safe / scale, 0.0, 50.0))

    # Stable zero-based ranks from each source and into each destination.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    ranks = np.arange(n, dtype=float)
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]

    row_rank[rows, row_order] = ranks[None, :]
    col_rank[col_order, cols] = ranks[:, None]

    # Exponential rank decay emphasizes the best alternatives.
    row_rank_score = np.exp(-row_rank / max(1.0, np.sqrt(n)))
    col_rank_score = np.exp(-col_rank / max(1.0, np.sqrt(n)))
    rank_score = np.sqrt(row_rank_score * col_rank_score)

    # Relative local compatibility with the best option at each endpoint.
    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    row_gap = np.clip((safe - row_best[:, None]) / scale, 0.0, 50.0)
    col_gap = np.clip((safe - col_best[None, :]) / scale, 0.0, 50.0)
    compatibility = np.exp(-0.5 * (row_gap + col_gap))

    # Mutual-nearest-neighbor bonus.
    nearest_row = np.argmin(safe, axis=1)
    nearest_col = np.argmin(safe, axis=0)

    mutual = np.zeros((n, n), dtype=float)
    source = np.arange(n)
    destinations = nearest_row
    is_finite = np.isfinite(safe[source, destinations])
    mutual[source[is_finite], destinations[is_finite]] = (
        nearest_col[destinations[is_finite]] == source[is_finite]
    )

    score = (
        distance_score
        * (0.45 + 0.55 * rank_score)
        * (0.65 + 0.35 * compatibility)
        * (1.0 + 0.15 * mutual)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
