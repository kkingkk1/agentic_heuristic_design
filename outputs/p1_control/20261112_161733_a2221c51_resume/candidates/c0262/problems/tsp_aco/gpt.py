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

    # Use the second-nearest edge as a less suppressive local density scale.
    second_index = min(1, n - 2)
    local_scale = sorted_edges[:, second_index]

    finite_local = np.isfinite(local_scale)
    if np.any(~finite_local):
        for i in np.flatnonzero(~finite_local):
            values = d[i, positive[i]]
            local_scale[i] = (
                float(np.median(values))
                if values.size
                else fallback_distance
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

    # Softer penalties preserve useful long cross-density connections.
    local_preference = np.exp(-1.08 * relative_length)
    rank_preference = np.exp(-2.55 * mutual_rank)
    global_preference = np.exp(-0.22 * safe_distance / global_scale)

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    density_mismatch = scale_difference / (scale_sum + eps)
    balance_preference = 1.0 / (1.0 + 0.85 * density_mismatch)

    # A soft floor prevents density mismatch from eliminating all
    # cross-region edges, while retaining distance and rank discrimination.
    cross_density_floor = 0.18 + 0.82 * np.exp(-0.75 * density_mismatch)

    score = (
        local_preference
        * rank_preference
        * global_preference
        * balance_preference
        * cross_density_floor
    )

    valid_edges = positive & positive.T
    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
