import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)
    if n == 1:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)

    fallback = np.median(d[positive]) if np.any(positive) else 1.0
    clean = np.where(positive, d, fallback)
    np.fill_diagonal(clean, np.inf)

    # Robust global distance scale.
    scale = np.median(clean[np.isfinite(clean)])
    scale = max(float(scale), np.finfo(float).eps)

    # Absolute closeness.
    closeness = np.exp(-clean / scale)
    closeness[~np.isfinite(closeness)] = 0.0

    # Relative preference at each endpoint.
    row_order = np.argsort(clean, axis=1)
    row_rank = np.empty_like(row_order)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)
    rank_score = 1.0 / (1.0 + row_rank.astype(float))
    rank_score[np.arange(n), np.arange(n)] = 0.0

    # Endpoint regret: favor edges that are much better than alternatives.
    sorted_distances = np.sort(clean, axis=1)
    best = sorted_distances[:, 0]
    second_best = sorted_distances[:, 1] if n > 2 else sorted_distances[:, 0]
    regret = np.maximum(second_best - best, 0.0) / (second_best + scale)

    edge_regret = 0.5 * (regret[:, None] + regret[None, :])
    relative_preference = np.sqrt(rank_score * rank_score.T)

    score = closeness * (0.5 + 0.3 * relative_preference + 0.2 * edge_regret)

    # Make the indicator compatible with symmetric TSP instances.
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    # Row normalization keeps sampling pressure comparable across nodes.
    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > np.finfo(float).eps,
    )

    return score.astype(float, copy=False)
