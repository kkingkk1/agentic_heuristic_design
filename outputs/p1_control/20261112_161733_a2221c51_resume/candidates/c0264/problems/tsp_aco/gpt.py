import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = np.isfinite(d) & (d > 0)
    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[positive]

    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(positive, d, np.inf)

    sorted_edges = np.sort(safe, axis=1)
    second_index = min(1, n - 2)
    local_scale = sorted_edges[:, second_index]

    finite_scale = np.isfinite(local_scale)
    if np.any(~finite_scale):
        row_counts = np.sum(positive, axis=1)
        row_medians = np.full(n, global_scale, dtype=float)

        for i in np.flatnonzero(~finite_scale):
            values = d[i, positive[i]]
            if values.size:
                row_medians[i] = float(np.median(values))

        local_scale = np.where(
            finite_scale,
            local_scale,
            np.where(row_counts > 0, row_medians, fallback_distance),
        )

    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback_distance * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    relative_length = safe_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    order = np.argsort(safe_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank /= max(n - 1, 1)

    mutual_rank = rank + rank.T
    one_sided_rank = np.minimum(rank, rank.T)

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    density_mismatch = scale_difference / (scale_sum + eps)

    log_score = (
        -1.18 * relative_length
        -2.95 * mutual_rank
        -0.22 * (safe_distance / global_scale)
        -0.34 * density_mismatch
        -0.55 * one_sided_rank
    )

    score = np.exp(np.clip(log_score, -745.0, 0.0))

    valid_edges = positive & positive.T
    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
