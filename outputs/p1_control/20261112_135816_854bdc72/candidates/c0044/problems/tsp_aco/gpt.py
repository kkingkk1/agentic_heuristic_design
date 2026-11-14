import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n <= 1:
        return result

    if np.any(np.isfinite(d) & (d < 0)):
        raise ValueError("distances must be non-negative")

    valid = np.isfinite(d) & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    symmetric_d = 0.5 * (d + d.T)
    symmetric_valid = valid & valid.T

    positive = symmetric_d[symmetric_valid & (symmetric_d > 0)]
    base_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    eps = max(np.finfo(float).eps, base_scale * 1e-12)

    safe_d = np.where(symmetric_valid, np.maximum(symmetric_d, 0.0), np.inf)
    np.fill_diagonal(safe_d, np.inf)

    row_values = np.where(
        np.isfinite(safe_d) & (safe_d > 0),
        safe_d,
        np.nan,
    )
    local_scale = np.nanmedian(row_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        base_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    normalized_distance = safe_d / endpoint_scale

    closeness = np.exp(-np.minimum(normalized_distance, 60.0))

    inverse_distance = 1.0 / (safe_d + eps)
    row_max = np.max(
        np.where(np.isfinite(inverse_distance), inverse_distance, 0.0),
        axis=1,
        keepdims=True,
    )
    row_max = np.maximum(row_max, eps)
    endpoint_preference = inverse_distance / row_max
    mutual_preference = np.sqrt(
        endpoint_preference * endpoint_preference.T
    )

    ranks = np.argsort(
        np.argsort(
            np.where(np.isfinite(safe_d), safe_d, np.inf),
            axis=1,
        ),
        axis=1,
    )
    rank_preference = 1.0 - ranks / max(n - 1, 1)
    rank_preference = np.sqrt(
        np.maximum(rank_preference, 0.0)
        * np.maximum(rank_preference.T, 0.0)
    )

    competition = np.sum(
        np.where(
            np.isfinite(normalized_distance),
            np.exp(-np.minimum(normalized_distance, 30.0)),
            0.0,
        ),
        axis=1,
    )
    diversity = 1.0 / np.sqrt(
        np.maximum(competition[:, None] * competition[None, :], eps)
    )
    diversity /= max(float(np.max(diversity)), eps)

    score = (
        np.maximum(closeness, eps) ** 0.45
        * np.maximum(mutual_preference, eps) ** 0.30
        * np.maximum(rank_preference, eps) ** 0.15
        * np.maximum(diversity, eps) ** 0.10
    )

    score[~symmetric_valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
