import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(d)

    if not np.all(finite | diagonal):
        raise ValueError(
            "distance_matrix contains invalid non-finite off-diagonal values"
        )

    work = np.where(finite, np.maximum(d, 0.0), np.inf)
    np.fill_diagonal(work, np.inf)

    positive = work[np.isfinite(work) & (work > 0.0)]
    global_fallback = (
        float(np.median(positive)) if positive.size else 1.0
    )
    eps = np.finfo(float).eps * max(1.0, global_fallback)

    positive_work = np.where(work > 0.0, work, np.inf)
    k = max(1, min(n - 1, int(np.sqrt(n))))

    nearest = np.partition(positive_work, k - 1, axis=1)[:, :k]
    valid_nearest = np.isfinite(nearest)

    local_sum = np.sum(
        np.where(valid_nearest, nearest, 0.0),
        axis=1,
    )
    local_count = np.sum(valid_nearest, axis=1)

    local_scale = np.divide(
        local_sum,
        np.maximum(local_count, 1),
        dtype=float,
    )
    local_scale = np.where(
        (local_count > 0) & np.isfinite(local_scale),
        np.maximum(local_scale, eps),
        max(global_fallback, eps),
    )

    normalized = work / (
        np.sqrt(local_scale[:, None] * local_scale[None, :]) + eps
    )

    endpoint_preference = np.exp(-normalized)
    mutual_preference = np.sqrt(
        endpoint_preference * endpoint_preference.T
    )

    isolation = local_scale / (
        np.median(local_scale) + eps
    )
    isolation = np.clip(isolation, 0.25, 4.0)
    isolation /= np.max(isolation)

    relative_quality = np.minimum(
        work / (local_scale[:, None] + eps),
        work / (local_scale[None, :] + eps),
    )
    bridge_bonus = 1.0 + 0.30 * np.maximum(
        isolation[:, None],
        isolation[None, :],
    ) * np.exp(-relative_quality)

    score = (
        endpoint_preference
        * (0.45 + 0.55 * mutual_preference)
        * bridge_bonus
    )

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
