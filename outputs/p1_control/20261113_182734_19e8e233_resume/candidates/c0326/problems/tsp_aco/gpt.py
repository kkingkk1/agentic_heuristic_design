import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return rank-based local-cycle priors for directed TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    masked = np.where(valid, d, np.inf)

    row_order = np.argsort(masked, axis=1, kind="stable")
    col_order = np.argsort(masked, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    positions = np.arange(n, dtype=float)
    row_rank[row_order, np.arange(n)[:, None]] = positions[None, :]
    col_rank[np.arange(n)[:, None], col_order] = positions[:, None]

    row_rank[~valid] = np.inf
    col_rank[~valid] = np.inf

    row_counts = valid.sum(axis=1)
    col_counts = valid.sum(axis=0)

    row_denominator = np.maximum(row_counts[:, None] - 1, 1)
    col_denominator = np.maximum(col_counts[None, :] - 1, 1)

    row_quality = 1.0 - row_rank / row_denominator
    col_quality = 1.0 - col_rank / col_denominator

    row_quality = np.clip(row_quality, 0.0, 1.0)
    col_quality = np.clip(col_quality, 0.0, 1.0)

    incoming_best = np.min(masked, axis=0)
    outgoing_best = np.min(masked, axis=1)

    incoming_best = np.where(np.isfinite(incoming_best), incoming_best, 0.0)
    outgoing_best = np.where(np.isfinite(outgoing_best), outgoing_best, 0.0)

    edge_scale = np.median(d[valid])
    edge_scale = max(float(edge_scale), eps)

    continuation_cost = d + outgoing_best[None, :]
    predecessor_cost = incoming_best[:, None] + d

    local_scale = edge_scale + np.median(d[valid])
    local_scale = max(float(local_scale), eps)

    continuation_bonus = 1.0 / (
        1.0 + np.maximum(continuation_cost, 0.0) / local_scale
    )
    predecessor_bonus = 1.0 / (
        1.0 + np.maximum(predecessor_cost, 0.0) / local_scale
    )

    reverse_distance = d.T
    reverse_valid = valid.T
    reciprocal = np.zeros_like(d, dtype=float)

    reciprocal[valid] = 1.0 / (
        1.0 + np.maximum(
            d[valid] + reverse_distance[valid],
            0.0,
        ) / local_scale
    )

    score = (
        (0.25 + 0.75 * row_quality)
        * (0.25 + 0.75 * col_quality)
        * (0.5 + 0.5 * continuation_bonus)
        * (0.5 + 0.5 * predecessor_bonus)
        * (0.75 + 0.25 * reciprocal)
    )

    score[~valid] = 0.0
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
