import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    eps = np.finfo(float).eps
    global_scale = max(float(np.median(positive)), eps)
    fallback = max(float(np.max(positive)), global_scale)
    replacement = max(fallback * 10.0, global_scale)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(3, n - 1)
    nearest = np.partition(work, neighbor_count - 1, axis=1)[:, :neighbor_count]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_length = work / np.maximum(geometric_scale, eps)
    absolute_length = work / global_scale

    neighbor_threshold = np.max(nearest, axis=1)
    neighbor_mask = work <= neighbor_threshold[:, None]
    mutual_neighbor = neighbor_mask & neighbor_mask.T

    rank_proxy = normalized_length / np.maximum(
        np.mean(np.where(neighbor_mask, normalized_length, 0.0), axis=1)[:, None],
        eps,
    )
    rank_signal = 1.0 / (1.0 + rank_proxy)

    local_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    local_gap /= 1.0 + np.log1p(
        np.maximum(local_scale[:, None], eps)
        + np.maximum(local_scale[None, :], eps)
    )

    score = (
        0.62 / (1.0 + normalized_length)
        + 0.18 / (1.0 + absolute_length)
        + 0.16 * rank_signal
        + 0.12 * mutual_neighbor.astype(float)
    )
    score *= 1.0 / (1.0 + 0.35 * local_gap)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
