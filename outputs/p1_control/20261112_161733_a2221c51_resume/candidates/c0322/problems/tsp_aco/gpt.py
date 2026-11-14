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

    positive = np.isfinite(d) & (d > 0.0)
    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[positive]

    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(positive, d, np.inf)
    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback_distance * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    # Obtain the second finite neighbor with partial selection.
    if n > 2:
        local_scale = np.partition(safe_distance, 1, axis=1)[:, 1]
    else:
        local_scale = safe_distance[:, 0]

    row_counts = np.sum(positive, axis=1)
    invalid_scale = ~np.isfinite(local_scale)
    if np.any(invalid_scale):
        row_medians = np.full(n, global_scale, dtype=float)
        for i in np.flatnonzero(invalid_scale):
            values = d[i, positive[i]]
            if values.size:
                row_medians[i] = float(np.median(values))

        local_scale = np.where(
            invalid_scale,
            np.where(row_counts > 0, row_medians, fallback_distance),
            local_scale,
        )

    local_scale = np.maximum(local_scale, eps)

    relative_length = safe_distance / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    # One stable ordering supplies the mutual-rank signal.
    order = np.argsort(safe_distance, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank /= max(n - 1, 1.0)

    mutual_rank = 0.5 * (rank + rank.T)

    # Main global signal: favor short edges without eliminating longer bridges.
    global_short = 1.0 / (
        1.0 + safe_distance / global_scale
    )

    # Local signal: reward edges that are short for both endpoints.
    local_short = 1.0 / (
        1.0 + relative_length
    )

    # Reciprocal neighborhood preference with moderate concentration.
    reciprocal_rank = np.exp(-2.25 * mutual_rank)

    # Mild density-balance correction. It preserves cross-density edges
    # through a soft floor instead of applying a hard penalty.
    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    density_balance = 1.0 / (
        1.0 + scale_difference / (scale_sum + eps)
    )
    density_balance = 0.86 + 0.14 * density_balance

    # Weighted combination avoids multiplicative over-suppression.
    score = (
        0.38 * global_short
        + 0.30 * local_short
        + 0.22 * reciprocal_rank
        + 0.10 * density_balance
    )

    score[~(positive & positive.T)] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
