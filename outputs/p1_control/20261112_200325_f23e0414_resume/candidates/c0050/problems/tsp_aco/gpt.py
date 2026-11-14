import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & (d > 0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    positive = d[valid]
    fallback = float(np.max(positive))
    scale = float(np.median(positive))
    eps = np.finfo(float).eps * max(scale, 1.0)

    safe = np.where(valid, d, fallback)

    inverse = np.zeros_like(d, dtype=float)
    inverse[valid] = 1.0 / (safe[valid] + eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    normalized_inverse = np.divide(
        inverse,
        row_max,
        out=np.zeros_like(inverse),
        where=row_max > 0,
    )

    row_sum = np.sum(inverse, axis=1, keepdims=True)
    row_preference = np.divide(
        inverse,
        row_sum,
        out=np.zeros_like(inverse),
        where=row_sum > 0,
    )

    local_min = np.min(np.where(valid, safe, np.inf), axis=1)
    local_min = np.where(np.isfinite(local_min), local_min, fallback)

    relative_gap = np.maximum(safe - local_min[:, None], 0.0)
    local_affinity = np.exp(-relative_gap / (local_min[:, None] + scale + eps))
    local_affinity[~valid] = 0.0

    sorted_distances = np.sort(np.where(valid, safe, np.inf), axis=1)
    nearest = sorted_distances[:, 0]
    second_nearest = sorted_distances[:, 1]

    nearest = np.where(np.isfinite(nearest), nearest, fallback)
    second_nearest = np.where(
        np.isfinite(second_nearest),
        second_nearest,
        nearest,
    )

    regret = np.maximum(second_nearest - nearest, 0.0)
    positive_regret = regret[regret > 0]
    regret_scale = (
        float(np.median(positive_regret))
        if positive_regret.size
        else scale
    )
    regret_weight = 1.0 + regret / (regret_scale + eps)

    mutual_preference = np.sqrt(
        np.maximum(row_preference, 0.0)
        * np.maximum(row_preference.T, 0.0)
    )

    nearest_indices = np.argmin(
        np.where(valid, safe, np.inf),
        axis=1,
    )
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i, j in enumerate(nearest_indices):
        if valid[i, j] and nearest_indices[j] == i:
            mutual_nearest[i, j] = 1.0

    regret_bonus = np.sqrt(
        regret_weight[:, None] * regret_weight[None, :]
    )
    regret_bonus /= np.max(regret_bonus) if np.max(regret_bonus) > 0 else 1.0

    score = (
        0.50 * normalized_inverse
        + 0.20 * local_affinity
        + 0.20 * mutual_preference
        + 0.10 * mutual_nearest
    )
    score *= regret_bonus
    score[~valid] = 0.0

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
