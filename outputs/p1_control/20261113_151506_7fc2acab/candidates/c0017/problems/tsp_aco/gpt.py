import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = distance_matrix.copy()
    d[~np.isfinite(d)] = np.nan
    d = np.where(np.isnan(d), np.nanmedian(d), d)
    d = np.maximum(d, 0.0)

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(finite_values))
    d[~np.isfinite(d)] = fallback

    positive = d[d > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    inverse_distance = 1.0 / (d + scale * 1e-6)
    inverse_distance /= np.max(inverse_distance)

    order = np.argsort(d, axis=1)
    nearest_k = max(1, min(3, n - 1))

    nearest_mask = np.zeros((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    nearest_mask[rows, order[:, :nearest_k]] = 1.0

    mutual_support = nearest_mask * nearest_mask.T
    one_sided_support = 0.5 * (nearest_mask + nearest_mask.T)

    local_scale = np.partition(d, nearest_k - 1, axis=1)[:, nearest_k - 1]
    local_scale = np.maximum(local_scale, scale * 1e-6)

    relative_length = d / np.sqrt(
        local_scale[:, None] * local_scale[None, :] + scale * scale * 1e-12
    )
    structural_quality = np.exp(-0.5 * np.maximum(relative_length - 1.0, 0.0) ** 2)

    score = (
        0.58 * inverse_distance
        + 0.24 * structural_quality
        + 0.12 * one_sided_support
        + 0.06 * mutual_support
    )

    np.fill_diagonal(score, 0.0)

    score_min = np.min(score)
    score_max = np.max(score)
    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    else:
        score.fill(0.0)

    np.fill_diagonal(score, 0.0)
    return score
