import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = np.maximum((d + d.T) * 0.5, 0.0)
    result = np.zeros_like(d, dtype=float)

    if n == 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_offdiag = np.isfinite(d) & ~diagonal
    positive = d[finite_offdiag & (d > 0)]

    if positive.size == 0:
        return result

    scale = float(np.median(positive))
    eps = np.finfo(float).eps * max(scale, 1.0)

    safe_d = np.where(np.isfinite(d), d, np.inf)
    safe_d = np.where(diagonal, np.inf, safe_d)

    inverse_score = 1.0 / (safe_d + eps)
    inverse_score[~np.isfinite(inverse_score)] = 0.0
    inverse_max = np.max(inverse_score)

    if inverse_max > 0 and np.isfinite(inverse_max):
        inverse_score /= inverse_max

    row_sorted = np.sort(safe_d, axis=1)
    nearest = row_sorted[:, 0]
    second_nearest = row_sorted[:, min(1, n - 2)]

    local_scale = np.sqrt(
        np.maximum(nearest, eps) * np.maximum(second_nearest, eps)
    )

    local_score = local_scale[:, None] / (
        safe_d + 0.75 * local_scale[None, :] + eps
    )
    local_score[~np.isfinite(local_score)] = 0.0

    ranks = np.argsort(np.argsort(safe_d, axis=1), axis=1).astype(float)
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~np.isfinite(rank_score)] = 0.0

    symmetric_rank_score = np.sqrt(rank_score * rank_score.T)

    score = (
        0.58 * inverse_score
        + 0.24 * local_score
        + 0.18 * symmetric_rank_score
    )

    score = (score + score.T) * 0.5
    score[~finite_offdiag] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
