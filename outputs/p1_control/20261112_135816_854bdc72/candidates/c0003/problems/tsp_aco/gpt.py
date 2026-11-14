import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d, dtype=float)

    clean = np.where(finite, d, np.nan)
    clean = np.where(np.isnan(clean), np.nanmax(clean), clean)
    clean = np.maximum(clean, 0.0)
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, 0.0)

    work = clean.copy()
    np.fill_diagonal(work, np.inf)

    order = np.argsort(work, axis=1, kind="mergesort")
    sorted_dist = np.take_along_axis(work, order, axis=1)

    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    local_scale = np.median(sorted_dist[:, :k], axis=1)
    fallback_scale = np.median(sorted_dist[np.isfinite(sorted_dist)])
    if not np.isfinite(fallback_scale) or fallback_scale <= 0:
        fallback_scale = 1.0
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        fallback_scale,
    )

    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(1, n)
    rank_norm = ranks / max(1, n - 1)

    row_scale = local_scale[:, None]
    pair_scale = 0.5 * (row_scale + row_scale.T)

    compactness = np.exp(-clean / (pair_scale + 1e-12))

    rank_support = np.exp(
        -1.35 * (rank_norm + rank_norm.T)
    )

    alternative = np.minimum(
        sorted_dist[:, min(1, n - 2)][:, None],
        sorted_dist[:, min(1, n - 2)][None, :],
    )
    scarcity = 1.0 / (
        1.0 + np.exp(np.clip((clean - alternative) / (pair_scale + 1e-12), -40.0, 40.0))
    )

    positive = clean[clean > 0]
    global_scale = np.median(positive) if positive.size else fallback_scale
    global_decay = np.exp(-clean / (global_scale + 1e-12))

    score = (
        0.40 * rank_support
        + 0.28 * compactness
        + 0.20 * scarcity
        + 0.12 * global_decay
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
