import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros_like(d, dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(d) & (d >= 0)
    off_diagonal = ~np.eye(n, dtype=bool)
    usable = valid & off_diagonal

    positive = d[usable & (d > 0)]
    eps = max(float(np.median(positive)) * 1e-12, 1e-12) if positive.size else 1e-12

    safe_d = np.where(usable, d, np.inf)
    safe_d = np.where((safe_d > 0) & np.isfinite(safe_d), safe_d, eps)

    row_scale = np.median(
        np.where(usable, d, np.nan),
        axis=1
    )
    fallback_scale = np.median(positive) if positive.size else 1.0
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        fallback_scale
    )
    row_scale = np.maximum(row_scale, eps)

    closeness = np.exp(-safe_d / row_scale[:, None])

    order = np.argsort(safe_d, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)

    endpoint_preference = np.sqrt(rank_score * rank_score.T)

    sparsity = np.sqrt(row_scale[:, None] * row_scale[None, :])
    sparsity = sparsity / (sparsity.max() + eps)

    score = (
        0.55 * closeness
        + 0.25 * endpoint_preference
        + 0.20 * sparsity
    )

    score = np.where(usable, score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = score.max()
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
