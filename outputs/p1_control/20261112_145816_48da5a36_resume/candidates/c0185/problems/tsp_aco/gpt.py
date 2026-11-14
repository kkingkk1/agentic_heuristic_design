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
    fallback = max(float(np.max(positive)), eps)
    replacement = max(8.0 * fallback, eps)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = max(1, min(3, n - 1))
    local_scale = np.partition(
        work,
        neighbor_count - 1,
        axis=1,
    )[:, neighbor_count - 1]
    local_scale = np.maximum(local_scale, eps)

    relative_length = work / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    endpoint_affinity = np.sqrt(
        (local_scale[:, None] / (work + eps))
        * (local_scale[None, :] / (work + eps))
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    mutual_rank = 1.0 / (1.0 + ranks + ranks.T)

    global_scale = max(float(np.median(positive)), eps)
    global_length = work / global_scale

    scale_mismatch = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )

    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )

    log_score = (
        1.25 * np.log(endpoint_affinity + eps)
        + 1.45 * np.log(mutual_rank)
        - 0.60 * relative_length
        - 0.18 * global_length
        - 0.80 * scale_mismatch
        - 0.40 * np.maximum(endpoint_longness - 1.0, 0.0)
    )

    finite_scores = log_score[valid]
    log_score -= np.max(finite_scores)

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
