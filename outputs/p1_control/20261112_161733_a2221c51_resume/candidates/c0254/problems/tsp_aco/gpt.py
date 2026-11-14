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

    valid = np.isfinite(d) & (d > 0)
    valid_edges = valid & valid.T
    if not np.any(valid_edges):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(valid, d, np.inf)

    kth = min(1, n - 2)
    local_scale = np.partition(safe, kth=kth, axis=1)[:, kth]
    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.where(
        np.isfinite(safe),
        safe,
        fallback_distance * 10.0,
    )
    safe_distance = np.maximum(safe_distance, eps)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_length = safe_distance / np.sqrt(
            local_scale[:, None] * local_scale[None, :]
        )
        endpoint_preference = np.exp(-1.10 * relative_length)

        global_preference = 1.0 / (
            1.0 + 0.35 * safe_distance / global_scale
        )
        global_preference = 0.62 + 0.38 * global_preference

        scale_difference = np.abs(
            local_scale[:, None] - local_scale[None, :]
        )
        scale_sum = local_scale[:, None] + local_scale[None, :]
        raw_balance = 1.0 / (
            1.0 + 0.85 * scale_difference / (scale_sum + eps)
        )
        neighborhood_balance = 0.78 + 0.22 * raw_balance

        score = (
            endpoint_preference
            * global_preference
            * neighborhood_balance
        )

    nearest = np.argmin(safe, axis=1)
    rows = np.arange(n)
    nearest_valid = np.isfinite(safe[rows, nearest])

    mutual = np.zeros((n, n), dtype=bool)
    mutual[rows[nearest_valid], nearest[nearest_valid]] = True
    mutual = mutual & mutual.T & valid_edges

    score *= np.where(mutual, 1.04, 1.0)
    score[~valid_edges] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
