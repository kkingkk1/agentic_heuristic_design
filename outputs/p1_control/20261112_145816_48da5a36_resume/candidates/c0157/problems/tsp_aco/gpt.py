import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = 0.5 * (d + d.T)
    eps = np.finfo(float).eps

    valid = np.isfinite(d) & (d > 0.0)
    valid &= valid.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive_values = d[valid]
    global_scale = max(float(np.median(positive_values)), eps)

    normalized = np.zeros((n, n), dtype=float)
    normalized[valid] = d[valid] / global_scale
    normalized = np.minimum(normalized, 1e12)

    finite_fallback = max(float(np.max(normalized[valid])), 1.0)
    invalid_value = min(finite_fallback * 16.0, 1e12)

    work = np.full((n, n), invalid_value, dtype=float)
    work[valid] = normalized[valid]
    np.fill_diagonal(work, invalid_value)

    valid_count = np.sum(valid, axis=1).astype(float)
    row_sum = np.sum(np.where(valid, normalized, 0.0), axis=1)
    local_scale = row_sum / np.maximum(valid_count, 1.0)
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_length = normalized / np.maximum(endpoint_scale, eps)

    accessibility = np.exp(
        -np.minimum(relative_length, 60.0)
    ) * valid

    accessibility_mass = np.sum(accessibility, axis=1)
    accessibility_mass = np.maximum(accessibility_mass, eps)

    exclusivity = accessibility / accessibility_mass[:, None]
    mutual_exclusivity = np.sqrt(
        np.maximum(exclusivity, eps)
        * np.maximum(exclusivity.T, eps)
    )

    two_hop = np.min(
        work[:, :, None] + work[None, :, :],
        axis=1,
    )
    two_hop = np.minimum(two_hop, 1e12)

    direct = np.maximum(normalized, eps)
    detour_advantage = (
        np.log1p(two_hop) - np.log1p(direct)
    )
    detour_advantage = np.clip(detour_advantage, -20.0, 20.0)

    local_gap = np.abs(
        np.log(np.maximum(local_scale[:, None], eps))
        - np.log(np.maximum(local_scale[None, :], eps))
    )
    local_gap = np.minimum(local_gap, 20.0)

    endpoint_mismatch = np.abs(
        np.log(np.maximum(relative_length, eps))
    )
    endpoint_mismatch = np.minimum(endpoint_mismatch, 20.0)

    log_score = (
        1.65 * np.log(np.maximum(mutual_exclusivity, eps))
        + 1.10 * np.tanh(detour_advantage)
        - 0.28 * local_gap
        - 0.18 * endpoint_mismatch
    )

    log_score = np.where(valid, log_score, -np.inf)

    finite_scores = np.isfinite(log_score)
    if not np.any(finite_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(log_score[finite_scores]))
    score = np.exp(
        np.clip(log_score - maximum, -745.0, 0.0)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
