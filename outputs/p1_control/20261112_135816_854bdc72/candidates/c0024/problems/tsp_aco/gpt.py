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
    positive = d[finite & (d > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0

    work = np.where(finite, d, fallback * 1e6)
    work = np.maximum(work, 0.0)
    np.fill_diagonal(work, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)

    row_values = np.where(off_diagonal, work, np.nan)
    row_scale = np.nanmedian(row_values, axis=1)
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0),
        row_scale,
        fallback,
    )

    closeness = np.exp(
        -work / np.sqrt(row_scale[:, None] * row_scale[None, :])
    )

    rank = np.zeros((n, n), dtype=float)
    for i in range(n):
        rank[i] = np.sum(work[i, :, None] < work[i, None, :], axis=0)
    rank_score = 1.0 / (1.0 + rank / max(n - 1, 1))

    detour = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        candidate = work[:, k, None] + work[None, k, :]
        detour = np.minimum(detour, candidate)

    scale = np.median(positive) if positive.size else fallback
    advantage = np.maximum(detour - work, 0.0)
    detour_score = 1.0 - np.exp(-advantage / max(scale, np.finfo(float).eps))

    score = (
        0.45 * closeness
        + 0.30 * rank_score
        + 0.25 * detour_score
    )

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
