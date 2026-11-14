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

    positive = d[valid]
    epsilon = np.finfo(float).eps
    fallback = float(np.median(positive))

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    safe_d = np.where(valid, d, fallback * 10.0)
    safe_d = np.maximum(safe_d, epsilon)

    local_scale = np.partition(safe_d, 1, axis=1)[:, 1]
    local_scale = np.maximum(local_scale, epsilon)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = d / np.maximum(geometric_scale, epsilon)

    short_edge_score = np.exp(-0.85 * normalized_distance)

    valid_count = np.maximum(valid.sum(axis=1), 1)
    neighbor_limit = np.maximum(
        3,
        np.ceil(0.20 * valid_count).astype(int),
    )
    neighbor_limit = np.minimum(neighbor_limit, n - 1)

    max_neighbors = int(np.max(neighbor_limit))
    near_endpoint = np.zeros((n, n), dtype=bool)

    if max_neighbors > 0:
        nearest_indices = np.argpartition(
            safe_d,
            kth=max_neighbors - 1,
            axis=1,
        )[:, :max_neighbors]

        rows = np.arange(n)[:, None]
        near_endpoint[rows, nearest_indices] = True
        near_endpoint &= valid

    at_least_one_near = near_endpoint | near_endpoint.T

    short_and_promising = (
        (normalized_distance <= 2.25)
        & at_least_one_near
        & valid
    )

    scale_ratio = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + epsilon
    )

    penalty_strength = np.where(
        short_and_promising,
        0.16,
        0.52,
    )
    scale_compatibility = np.exp(-penalty_strength * scale_ratio)

    score = short_edge_score * scale_compatibility
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
