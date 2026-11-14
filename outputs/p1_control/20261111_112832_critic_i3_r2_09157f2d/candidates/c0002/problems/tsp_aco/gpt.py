import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    x = d.copy()
    np.fill_diagonal(x, np.nan)

    positive = x[np.isfinite(x) & (x > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    x[~np.isfinite(x)] = fallback
    x = np.maximum(x, 0.0)

    off_diag = ~np.eye(n, dtype=bool)

    row_scale = np.median(np.where(off_diag, x, np.nan), axis=1)
    row_scale = np.where(row_scale > 0, row_scale, fallback)

    closeness = np.exp(-x / (row_scale[:, None] + 1e-12))
    closeness[~off_diag] = 0.0

    row_best = np.min(np.where(off_diag, x, np.inf), axis=1)
    row_best = np.where(np.isfinite(row_best), row_best, fallback)

    relative = row_best[:, None] / (x + 1e-12)
    relative = np.clip(relative, 0.0, 1.0)
    relative[~off_diag] = 0.0

    nearest = np.argmin(np.where(off_diag, x, np.inf), axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if nearest[j] == i:
            mutual[i, j] = 1.0

    tri_scale = np.median(x[off_diag]) + 1e-12
    triangle_support = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            via_k = x[i, :, None] + x[:, j][None, :]
            via_k[i, :] = np.inf
            via_k[:, j] = np.inf
            best_two_step = np.min(via_k)
            triangle_support[i, j] = np.exp(-best_two_step / tri_scale)

    score = (
        0.50 * closeness
        + 0.25 * relative
        + 0.15 * triangle_support
        + 0.10 * mutual
    )

    score[~off_diag] = 0.0
    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
