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
        return np.zeros((n, n), dtype=float)

    positive = d[valid]
    eps = np.finfo(float).eps
    replacement = max(10.0 * float(np.max(positive)), eps)

    work = np.where(valid, d, replacement)
    np.fill_diagonal(work, replacement)

    neighbor_count = min(2, n - 1)
    local_scale = np.partition(
        work,
        neighbor_count - 1,
        axis=1,
    )[:, neighbor_count - 1]
    local_scale = np.maximum(local_scale, eps)

    log_work = np.log(np.maximum(work, eps))
    log_local = np.log(local_scale)

    geometric_scale = np.exp(
        0.5 * (log_local[:, None] + log_local[None, :])
    )

    relative_length = work / geometric_scale

    endpoint_affinity_log = (
        0.5 * (log_local[:, None] + log_local[None, :])
        - log_work
    )

    reciprocal_rank_log = -np.log1p(relative_length)

    global_scale = max(float(np.median(positive)), eps)
    global_length = work / global_scale

    scale_mismatch = (
        np.abs(local_scale[:, None] - local_scale[None, :])
        / (
            local_scale[:, None]
            + local_scale[None, :]
            + eps
        )
    )

    endpoint_longness = np.maximum(
        work / local_scale[:, None],
        work / local_scale[None, :],
    )

    with np.errstate(over="ignore", invalid="ignore"):
        log_score = (
            1.40 * endpoint_affinity_log
            + 1.50 * reciprocal_rank_log
            - 0.60 * relative_length
            - 0.14 * global_length
            - 0.65 * scale_mismatch
            - 0.30 * np.maximum(endpoint_longness - 1.0, 0.0)
        )

    finite_scores = log_score[valid]
    shift = float(np.max(finite_scores))
    log_score = log_score - shift

    with np.errstate(over="ignore", invalid="ignore"):
        score = np.exp(np.clip(log_score, -745.0, 0.0))

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
