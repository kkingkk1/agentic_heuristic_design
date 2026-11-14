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
    off_diag = ~np.eye(n, dtype=bool)
    valid = finite & off_diag

    safe_d = np.where(valid, np.maximum(d, 0.0), np.nan)

    row_scale = np.nanmedian(safe_d, axis=1)
    global_scale = np.nanmedian(safe_d)
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        global_scale,
    )

    relative = safe_d / np.sqrt(row_scale[:, None] * row_scale[None, :])
    relative = np.where(valid, relative, np.inf)

    ranks = np.argsort(np.argsort(relative, axis=1), axis=1)
    rank_score = np.exp(-ranks / max(1.0, 0.35 * (n - 1)))

    nearest = np.argmin(relative, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0

    local_score = np.exp(-np.minimum(relative, 20.0))
    priority = local_score * (0.65 + 0.35 * rank_score)
    priority *= 1.0 + 0.8 * mutual
    priority = np.where(valid, priority, 0.0)

    priority = 0.5 * (priority + priority.T)
    np.fill_diagonal(priority, 0.0)

    row_sums = priority.sum(axis=1, keepdims=True)
    fallback = np.where(valid, 1.0, 0.0)
    fallback_sums = fallback.sum(axis=1, keepdims=True)
    result = np.divide(
        priority,
        row_sums,
        out=np.zeros_like(priority),
        where=row_sums > 0,
    )
    result = np.where(
        row_sums > 0,
        result,
        np.divide(
            fallback,
            fallback_sums,
            out=np.zeros_like(fallback),
            where=fallback_sums > 0,
        ),
    )
    return result
