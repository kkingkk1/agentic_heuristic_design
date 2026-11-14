import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, scale-adaptive edge-prior matrix for TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0.0)
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid]
    global_scale = max(float(np.median(positive)), eps)
    fallback = max(float(np.max(positive)), global_scale)

    safe_distance = np.where(valid, d, 10.0 * fallback)
    safe_distance = np.maximum(safe_distance, eps)

    ranked_distance = np.where(valid, d, np.inf)
    finite_counts = np.sum(valid, axis=1)

    sorted_distance = np.sort(ranked_distance, axis=1)
    third_index = min(2, n - 2)
    local_scale = sorted_distance[:, third_index]

    local_scale = np.where(
        (finite_counts > third_index)
        & np.isfinite(local_scale)
        & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = safe_distance / endpoint_scale
    global_length = safe_distance / global_scale

    order = np.argsort(ranked_distance, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    ranks /= max(n - 1, 1)

    mutual_rank = 1.0 / (1.0 + ranks + ranks.T)

    endpoint_ratio = np.maximum(
        safe_distance / local_scale[:, None],
        safe_distance / local_scale[None, :],
    )

    scale_mismatch = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + eps
    )

    local_density = np.mean(
        valid
        & (
            safe_distance
            <= 1.5 * local_scale[:, None]
        ),
        axis=1,
    )

    density_penalty = (
        local_density[:, None] + local_density[None, :]
    ) * np.minimum(relative_length, 3.0)

    log_score = (
        -0.82 * relative_length
        -0.24 * global_length
        + 1.45 * np.log(mutual_rank)
        -0.38 * np.maximum(endpoint_ratio - 1.0, 0.0)
        -0.72 * scale_mismatch
        -0.14 * density_penalty
    )

    log_score = np.where(valid, log_score, -np.inf)
    finite_scores = np.isfinite(log_score)

    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    shift = float(np.max(log_score[finite_scores]))
    score = np.zeros((n, n), dtype=float)
    score[finite_scores] = np.exp(
        np.clip(log_score[finite_scores] - shift, -745.0, 0.0)
    )

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
