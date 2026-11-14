import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_positive = np.isfinite(d) & (d > 0)
    np.fill_diagonal(finite_positive, False)

    if not np.any(finite_positive):
        return np.zeros((n, n), dtype=float)

    d_clean = np.where(finite_positive, d, np.inf)

    both_valid = finite_positive & finite_positive.T
    one_valid = finite_positive | finite_positive.T

    symmetric_distance = np.full((n, n), np.inf, dtype=float)
    symmetric_distance[both_valid] = (
        d[both_valid] + d.T[both_valid]
    ) * 0.5
    symmetric_distance[~both_valid & finite_positive] = d[
        ~both_valid & finite_positive
    ]
    symmetric_distance[~both_valid & finite_positive.T] = d.T[
        ~both_valid & finite_positive.T
    ]
    d = symmetric_distance
    valid = one_valid

    positive_values = d[valid]
    global_scale = max(
        float(np.median(positive_values)),
        np.finfo(float).eps,
    )

    safe_d = np.where(valid, d, global_scale * 10.0)
    np.fill_diagonal(safe_d, global_scale * 10.0)
    safe_d = np.maximum(safe_d, np.finfo(float).eps)

    neighbor_count = min(2, n - 1)
    local_scale = np.partition(
        safe_d,
        neighbor_count - 1,
        axis=1,
    )[:, neighbor_count - 1]
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = d / geometric_scale

    absolute_score = np.exp(-d / global_scale)
    relative_score = np.exp(-0.65 * relative_distance)

    inverse_distance = 1.0 / (
        1.0 + d / global_scale
    )
    local_inverse = 1.0 / (
        1.0 + relative_distance
    )

    candidate_count = min(3, n - 1)
    candidate_indices = np.argpartition(
        safe_d,
        candidate_count - 1,
        axis=1,
    )[:, :candidate_count]

    candidate_mask = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    candidate_mask[rows, candidate_indices] = True
    np.fill_diagonal(candidate_mask, False)

    mutual = candidate_mask & candidate_mask.T
    one_sided = candidate_mask | candidate_mask.T

    reciprocity_bonus = (
        1.0
        + 0.38 * mutual.astype(float)
        + 0.08 * (one_sided & ~mutual).astype(float)
    )

    local_density = np.sum(
        np.exp(-safe_d / local_scale[:, None]) * valid,
        axis=1,
    )
    density_factor = 1.0 / np.sqrt(
        (1.0 + local_density[:, None])
        * (1.0 + local_density[None, :])
    )

    crowding_factor = 0.72 + 0.28 * density_factor / (
        np.max(density_factor) + np.finfo(float).eps
    )

    scale_ratio = np.abs(
        local_scale[:, None] - local_scale[None, :]
    ) / (
        local_scale[:, None]
        + local_scale[None, :]
        + np.finfo(float).eps
    )
    scale_factor = 0.78 + 0.22 * np.exp(-scale_ratio)

    score = (
        0.30 * absolute_score
        + 0.24 * relative_score
        + 0.24 * inverse_distance
        + 0.22 * local_inverse
    )

    score *= reciprocity_bonus
    score *= crowding_factor
    score *= scale_factor

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
