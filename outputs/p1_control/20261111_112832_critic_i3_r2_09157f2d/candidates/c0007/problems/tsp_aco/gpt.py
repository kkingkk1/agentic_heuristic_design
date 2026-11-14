import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = d[np.isfinite(d) & (d >= 0)]
    if finite_nonnegative.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(finite_nonnegative))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    clean = np.where(np.isfinite(d) & (d >= 0), d, fallback)
    clean = np.maximum(clean, 0.0)
    np.fill_diagonal(clean, np.inf)

    finite_edges = clean[np.isfinite(clean)]
    scale = float(np.median(finite_edges)) if finite_edges.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    inverse = np.zeros_like(clean)
    valid = np.isfinite(clean) & (clean > 0)
    inverse[valid] = scale / clean[valid]

    shortness = np.zeros_like(clean)
    shortness[valid] = np.exp(-clean[valid] / scale)

    ranks = np.full((n, n), float(n), dtype=float)
    for i in range(n):
        indices = np.flatnonzero(np.arange(n) != i)
        order = indices[np.argsort(clean[i, indices], kind="stable")]
        ranks[i, order] = np.arange(n - 1, dtype=float)

    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_score = np.exp(-ranks / rank_scale)
    rank_score[~np.isfinite(clean)] = 0.0

    nearest_indices = np.argmin(clean, axis=1)
    nearest = np.min(clean, axis=1)
    second_nearest = np.partition(clean, 1, axis=1)[:, 1]
    gaps = np.maximum(second_nearest - nearest, 0.0)

    competition = np.zeros_like(clean)
    for i in range(n):
        denominator = gaps[i] + 0.1 * scale
        valid_row = np.isfinite(clean[i])
        competition[i, valid_row] = np.exp(
            -(clean[i, valid_row] - nearest[i]) / denominator
        )

    mutual = np.zeros_like(clean)
    for i, j in enumerate(nearest_indices):
        if i != j and nearest_indices[j] == i:
            mutual[i, j] = 1.0

    score = (
        0.35 * inverse
        + 0.30 * shortness
        + 0.20 * rank_score
        + 0.10 * competition
        + 0.05 * mutual
    )

    score[~np.isfinite(score)] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
