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

    finite = np.isfinite(d)
    positive = d[(finite) & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0

    clean = np.where(finite & (d > 0), d, fallback)
    np.fill_diagonal(clean, np.inf)

    positive = clean[np.isfinite(clean)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inverse = 1.0 / (1.0 + clean / scale)
    inverse[~np.isfinite(inverse)] = 0.0

    row_order = np.argsort(clean, axis=1)
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + row_rank)

    col_order = np.argsort(clean, axis=0)
    col_rank = np.empty_like(col_order, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)
    rank_score = 0.5 * (rank_score + 1.0 / (1.0 + col_rank))

    nearest = row_order[:, 0]
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual[i, j] = 1.0

    prior = 0.60 * inverse + 0.30 * rank_score + 0.10 * mutual
    np.fill_diagonal(prior, 0.0)

    row_max = np.max(prior, axis=1, keepdims=True)
    prior = prior / np.maximum(row_max, np.finfo(float).eps)
    prior = np.sqrt(np.maximum(prior, 0.0))
    prior[~np.isfinite(prior)] = 0.0
    np.fill_diagonal(prior, 0.0)

    return prior.astype(float, copy=False)
