import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-prior indicators for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = d.copy()
    diagonal = np.eye(n, dtype=bool)
    off_diagonal = ~diagonal

    finite_off = np.isfinite(d) & off_diagonal
    if np.any(finite_off):
        finite_values = d[finite_off]
        scale = max(float(np.max(finite_values)), 1.0)
        replacement = scale * 10.0 + 1.0
    else:
        replacement = 1.0

    d[~np.isfinite(d) & off_diagonal] = replacement
    d[diagonal] = np.inf
    d = np.maximum(d, 0.0)

    row_scale = np.median(np.where(np.isfinite(d), d, np.nan), axis=1)
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 1e-12),
        row_scale,
        np.maximum(np.mean(np.where(np.isfinite(d), d, 0.0), axis=1), 1.0),
    )

    direct = np.exp(-d / (row_scale[:, None] + 1e-12))
    direct[diagonal] = 0.0

    row_order = np.argsort(d, axis=1)
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    rank_score = 1.0 - row_rank / max(n - 1, 1)
    rank_score[diagonal] = 0.0

    incoming_scale = np.median(np.where(np.isfinite(d), d, np.nan), axis=0)
    incoming_scale = np.where(
        np.isfinite(incoming_scale) & (incoming_scale > 1e-12),
        incoming_scale,
        np.maximum(np.mean(np.where(np.isfinite(d), d, 0.0), axis=0), 1.0),
    )

    reverse = np.exp(-d / (incoming_scale[None, :] + 1e-12))
    reverse[diagonal] = 0.0
    mutual = np.sqrt(direct * reverse)

    continuation = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            candidates = np.arange(n) != i
            candidates[j] = False
            if np.any(candidates):
                next_cost = np.min(d[j, candidates])
                continuation[i, j] = np.exp(
                    -next_cost / (row_scale[j] + 1e-12)
                )

    score = (
        0.45 * direct
        + 0.20 * rank_score
        + 0.20 * mutual
        + 0.15 * continuation
    )
    score[diagonal] = 0.0

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
