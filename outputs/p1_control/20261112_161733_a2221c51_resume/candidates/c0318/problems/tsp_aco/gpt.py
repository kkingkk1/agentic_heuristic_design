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
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive_values = d[valid]

    global_scale = max(float(np.median(positive_values)), eps)
    fallback_distance = max(float(np.max(positive_values)), global_scale)

    safe = np.where(valid, d, np.inf)
    safe_for_partition = np.where(
        np.isfinite(safe),
        safe,
        fallback_distance * 10.0,
    )

    neighbor_count = min(3, n - 1)
    local_scale = np.partition(
        safe_for_partition,
        kth=neighbor_count - 1,
        axis=1,
    )[:, neighbor_count - 1]

    local_scale = np.where(
        np.isfinite(local_scale),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    safe_distance = np.maximum(safe_for_partition, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_distance = safe_distance / pair_scale

        local_preference = np.exp(-1.20 * relative_distance)

        neighborhood_threshold = 1.35 * pair_scale
        neighborhood_ratio = safe_distance / np.maximum(
            neighborhood_threshold,
            eps,
        )

        neighborhood_preference = np.where(
            safe_distance <= neighborhood_threshold,
            0.72 + 0.28 * np.exp(-0.75 * neighborhood_ratio),
            0.28 + 0.12 * np.exp(-0.35 * neighborhood_ratio),
        )

        global_preference = 0.68 + 0.32 * np.exp(
            -0.24 * safe_distance / global_scale
        )

        scale_difference = np.abs(
            local_scale[:, None] - local_scale[None, :]
        )
        scale_sum = local_scale[:, None] + local_scale[None, :]
        balance = 1.0 / (
            1.0 + 0.90 * scale_difference / (scale_sum + eps)
        )
        balance_preference = 0.80 + 0.20 * balance

        score = (
            local_preference
            * neighborhood_preference
            * global_preference
            * balance_preference
        )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
