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
    safe_for_partition = np.where(
        np.isfinite(safe),
        np.maximum(safe, eps),
        fallback_distance * 10.0,
    )

    second_index = min(1, n - 1)
    local_scale = np.partition(
        safe_for_partition,
        second_index,
        axis=1,
    )[:, second_index]

    row_counts = np.sum(positive, axis=1)
    local_scale = np.where(
        row_counts > 0,
        local_scale,
        fallback_distance,
    )
    local_scale = np.maximum(local_scale, eps)

    relative_length = safe_for_partition / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    endpoint_preference = 1.0 / (1.0 + 1.15 * relative_length)

    global_preference = 0.72 + 0.28 / (
        1.0 + 0.30 * safe_for_partition / global_scale
    )

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None, :]
    neighborhood_balance = 0.80 + 0.20 / (
        1.0 + 0.85 * scale_difference / (scale_sum + eps)
    )

    candidate_count = min(
        n - 1,
        max(4, 2 * int(np.ceil(np.sqrt(n)))),
    )

    candidate_index = np.argpartition(
        safe_for_partition,
        candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate_mask[rows, candidate_index] = True
    np.fill_diagonal(candidate_mask, False)

    mutual_candidates = candidate_mask & candidate_mask.T
    candidate_bonus = 1.0 + 0.34 * candidate_mask + 0.18 * mutual_candidates

    score = (
        endpoint_preference
        * global_preference
        * neighborhood_balance
        * candidate_bonus
    )

    valid_edges = positive & positive.T
    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
