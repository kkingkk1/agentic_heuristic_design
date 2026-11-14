import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    valid = np.isfinite(d) & (d > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros_like(d, dtype=float)

    positive = d[valid]
    eps = np.finfo(float).eps
    replacement = max(10.0 * float(np.max(positive)), eps)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(2, n - 1)
    local_scale = np.partition(
        work,
        kth=neighbor_count - 1,
        axis=1,
    )[:, neighbor_count - 1]
    local_scale = np.maximum(local_scale, eps)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    safe_distance = work + eps

    relative_length = work / np.sqrt(row_scale * col_scale)

    endpoint_affinity = np.sqrt(
        (row_scale / safe_distance) * (col_scale / safe_distance)
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values
    mutual_rank = 1.0 / (1.0 + ranks + ranks.T)

    global_scale = max(float(np.median(positive)), eps)
    global_length = work / global_scale

    scale_mismatch = np.abs(row_scale - col_scale) / (
        row_scale + col_scale + eps
    )

    endpoint_longness = np.maximum(
        work / row_scale,
        work / col_scale,
    )

    log_score = (
        1.15 * np.log(endpoint_affinity + eps)
        + 1.35 * np.log(mutual_rank)
        - 0.55 * relative_length
        - 0.20 * global_length
        - 0.85 * scale_mismatch
        - 0.35 * np.maximum(endpoint_longness - 1.0, 0.0)
    )

    finite_log_score = log_score[valid]
    log_score -= np.max(finite_log_score)

    score = np.exp(np.clip(log_score, -745.0, 0.0))
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
