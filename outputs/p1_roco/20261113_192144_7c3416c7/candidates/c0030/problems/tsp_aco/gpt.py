import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)

    finite = np.isfinite(d)
    positive = finite & (d > 0)

    if not np.any(positive):
        return np.zeros_like(d, dtype=float)

    valid_values = d[positive]
    fallback = float(np.max(valid_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    safe = np.where(finite, np.maximum(d, 0.0), fallback * 10.0)
    safe = np.minimum(safe, fallback * 10.0)
    np.fill_diagonal(safe, fallback * 10.0)

    symmetric = 0.5 * (safe + safe.T)

    off_diagonal = ~np.eye(n, dtype=bool)
    row_scale = np.median(
        np.where(off_diagonal, symmetric, np.nan),
        axis=1
    )
    row_scale = np.where(np.isfinite(row_scale) & (row_scale > 0), row_scale, fallback)

    col_scale = np.median(
        np.where(off_diagonal, symmetric, np.nan),
        axis=0
    )
    col_scale = np.where(np.isfinite(col_scale) & (col_scale > 0), col_scale, fallback)

    local_ratio = symmetric / np.sqrt(row_scale[:, None] * col_scale[None, :])
    local_score = np.exp(-np.clip(local_ratio, 0.0, 50.0))

    inv_distance = 1.0 / (symmetric + np.finfo(float).eps)
    inv_distance /= np.max(inv_distance[off_diagonal])

    row_order = np.argsort(symmetric, axis=1)
    row_rank = np.empty_like(row_order, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(n, dtype=float)
    row_rank = 1.0 - row_rank / max(n - 1, 1)

    col_order = np.argsort(symmetric, axis=0)
    col_rank = np.empty_like(col_order, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n, dtype=float)[:, None]
    col_rank = 1.0 - col_rank / max(n - 1, 1)

    rank_score = 0.5 * (row_rank + col_rank)

    scores = (
        0.45 * inv_distance
        + 0.35 * local_score
        + 0.20 * rank_score
    )

    scores[~positive] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0 and np.isfinite(maximum):
        scores /= maximum

    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
