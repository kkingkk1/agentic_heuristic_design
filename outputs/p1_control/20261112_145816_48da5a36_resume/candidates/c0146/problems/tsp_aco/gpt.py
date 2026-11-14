import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = 0.5 * (d + d.T)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d > 0.0) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    eps = np.finfo(float).eps
    fallback = max(float(np.median(positive_values)), eps)

    safe_d = np.where(valid, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, eps)

    neighbor_index = min(2, n - 2)
    local_scale = np.partition(
        safe_d,
        neighbor_index,
        axis=1,
    )[:, neighbor_index]
    local_scale = np.maximum(local_scale, eps)

    row_ratio = safe_d / local_scale[:, None]
    col_ratio = safe_d / local_scale[None, :]

    normalized_distance = safe_d / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    distance_score = 1.0 / np.square(1.0 + normalized_distance)

    endpoint_score = (
        1.0 / (1.0 + row_ratio)
        * 1.0 / (1.0 + col_ratio)
    )

    scale_difference = np.abs(
        local_scale[:, None] - local_scale[None, :]
    )
    scale_sum = local_scale[:, None] + local_scale[None:] + eps
    scale_compatibility = 1.0 / (1.0 + scale_difference / scale_sum)

    neighborhood_weight = 1.0 / (1.0 + row_ratio)
    neighborhood_density = np.sum(
        neighborhood_weight * valid,
        axis=1,
    )
    crowding_penalty = 1.0 / np.sqrt(
        (1.0 + neighborhood_density[:, None])
        * (1.0 + neighborhood_density[None, :])
    )

    score = (
        distance_score
        * endpoint_score
        * scale_compatibility
        * crowding_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
