import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty_like(d, dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    if not np.any(finite_positive):
        return np.zeros_like(d, dtype=float)

    safe_d = d.copy()
    fallback = np.max(safe_d[finite_positive])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    safe_d[~finite_positive] = fallback

    np.fill_diagonal(safe_d, np.inf)

    scale = np.median(safe_d[np.isfinite(safe_d)])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    inverse = 1.0 / (safe_d + 1e-12)
    inverse[~np.isfinite(inverse)] = 0.0

    row_min = np.min(safe_d, axis=1, keepdims=True)
    row_min = np.where(np.isfinite(row_min), row_min, scale)
    mutual = np.exp(-0.5 * (safe_d / (row_min + scale * 1e-12)))
    mutual[~np.isfinite(mutual)] = 0.0

    local_scale = np.median(safe_d, axis=1, keepdims=True)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        scale,
    )
    neighborhood = np.exp(-safe_d / (local_scale + 1e-12))
    neighborhood[~np.isfinite(neighborhood)] = 0.0

    def normalize(x):
        x = np.maximum(x, 0.0)
        lo = np.min(x, axis=1, keepdims=True)
        hi = np.max(x, axis=1, keepdims=True)
        return (x - lo) / (hi - lo + 1e-12)

    score = (
        0.50 * normalize(inverse)
        + 0.30 * normalize(mutual)
        + 0.20 * normalize(neighborhood)
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    positive = score[score > 0]
    if positive.size:
        temperature = np.median(positive)
        score = np.power(score / (temperature + 1e-12), 1.15)

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
