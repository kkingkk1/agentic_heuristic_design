import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed TSP priors based on local path extensibility and ranks."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    safe = np.where(valid, d, np.inf)

    positive = d[valid & (d > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        finite = d[valid]
        scale = float(np.median(finite)) if finite.size else 1.0
    scale = max(scale, eps)

    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    row_has_edge = np.any(valid, axis=1)
    col_has_edge = np.any(valid, axis=0)

    row_best = np.where(row_has_edge, row_best, 0.0)
    col_best = np.where(col_has_edge, col_best, 0.0)

    outgoing_extension = row_best.copy()
    incoming_extension = col_best.copy()

    row_rank = np.full((n, n), float(n), dtype=float)
    col_rank = np.full((n, n), float(n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = indices[np.argsort(d[i, indices], kind="stable")]
            row_rank[i, order] = np.arange(1, order.size + 1, dtype=float)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = indices[
                np.argsort(d[indices, j], kind="stable")
            ]
            col_rank[order, j] = np.arange(1, order.size + 1, dtype=float)

    row_counts = valid.sum(axis=1).astype(float)
    col_counts = valid.sum(axis=0).astype(float)

    row_rank_norm = np.divide(
        row_rank,
        np.maximum(row_counts[:, None], 1.0),
    )
    col_rank_norm = np.divide(
        col_rank,
        np.maximum(col_counts[None, :], 1.0),
    )

    continuation = outgoing_extension[None, :]
    predecessor = incoming_extension[:, None]

    path_cost = d + 0.65 * continuation + 0.65 * predecessor
    normalized_cost = np.divide(path_cost, scale)

    rank_penalty = (
        0.55 * np.sqrt(row_rank_norm)
        + 0.45 * np.sqrt(col_rank_norm)
    )

    score = np.divide(
        1.0,
        1.0 + np.maximum(normalized_cost, 0.0),
    )
    score *= np.divide(1.0, 1.0 + rank_penalty)

    candidate_width = max(2, int(np.ceil(np.sqrt(max(n - 1, 1)))))
    local_candidate = (
        (row_rank <= candidate_width)
        | (col_rank <= candidate_width)
    )
    score *= np.where(local_candidate, 1.0, 0.35)

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
