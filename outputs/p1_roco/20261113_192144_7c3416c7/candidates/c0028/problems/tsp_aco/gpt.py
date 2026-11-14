import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    positive = finite & (d > 0)

    if not np.any(positive):
        result = np.zeros((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    positive_values = d[positive]
    scale = np.median(positive_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(positive_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_d = np.where(finite, d, scale * 1e6)
    safe_d = np.maximum(safe_d, scale * 1e-12)

    inverse_score = 1.0 / safe_d
    inverse_score /= np.max(inverse_score)

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1).astype(float)
    valid_counts = np.sum(finite, axis=1, keepdims=True)
    valid_counts = np.maximum(valid_counts, 1)
    rank_score = 1.0 - ranks / valid_counts
    rank_score = np.clip(rank_score, 0.0, 1.0)

    local_scale = np.median(
        np.where(finite, safe_d, np.nan),
        axis=1,
        keepdims=True
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        scale
    )

    relative_length = safe_d / np.sqrt(local_scale * local_scale.T)
    geometric_score = np.exp(-np.minimum(relative_length, 50.0))

    score = (
        0.50 * inverse_score
        + 0.35 * rank_score
        + 0.15 * geometric_score
    )

    score[~finite] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
