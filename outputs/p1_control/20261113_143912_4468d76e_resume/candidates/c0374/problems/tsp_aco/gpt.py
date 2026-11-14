import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(
        d,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    positive = np.isfinite(d) & (d > 0)
    valid_edges = positive & positive.T

    if not np.any(valid_edges):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[positive]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(positive, d, np.inf)
    sorted_edges = np.sort(safe, axis=1)

    second_index = min(1, n - 2)
    local_scale = sorted_edges[:, second_index]

    finite_local = np.isfinite(local_scale)
    if np.any(~finite_local):
        row_counts = np.sum(positive, axis=1)
        row_medians = np.full(n, global_scale, dtype=float)

        for i in np.flatnonzero(~finite_local):
            row_values = d[i, positive[i]]
            if row_values.size:
                row_medians[i] = float(np.median(row_values))

        local_scale = np.where(
            finite_local,
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

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_length = safe_distance / np.maximum(geometric_scale, eps)

    local_closeness = 1.0 / (1.0 + np.power(relative_length, 1.15))

    order = np.argsort(safe_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank /= max(n - 1, 1)

    mutual_rank = rank + rank.T
    rank_preference = 1.0 / (1.0 + 1.8 * mutual_rank)

    global_preference = 0.84 + 0.16 * np.exp(
        -0.18 * safe_distance / global_scale
    )

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    density_balance = 1.0 / (
        1.0 + 0.75 * scale_difference / (scale_sum + eps)
    )
    density_preference = 0.90 + 0.10 * density_balance

    score = (
        np.power(local_closeness, 0.68)
        * np.power(rank_preference, 0.32)
        * global_preference
        * density_preference
    )

    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
