import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    d = distance_matrix.copy()
    finite = np.isfinite(d)
    np.fill_diagonal(finite, False)

    if not np.any(finite):
        return scores

    positive = d[finite & (d > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps

    d[~np.isfinite(d)] = np.inf
    d[d < 0] = 0.0
    np.fill_diagonal(d, np.inf)

    finite_rows = np.isfinite(d)

    row_min = np.min(d, axis=1)
    row_min[~np.isfinite(row_min)] = fallback

    masked = np.where(finite_rows, d, np.nan)
    row_median = np.nanmedian(masked, axis=1)
    row_median[~np.isfinite(row_median)] = fallback

    local_scale = 0.5 * row_min + 0.5 * row_median
    local_scale = np.maximum(local_scale, eps)

    normalized = np.divide(
        d,
        local_scale[:, None],
        out=np.full_like(d, np.inf),
        where=finite_rows,
    )

    attraction = np.exp(-normalized)
    attraction[~finite_rows] = 0.0

    row_sum = attraction.sum(axis=1, keepdims=True)
    preference = np.divide(
        attraction,
        row_sum,
        out=np.zeros_like(attraction),
        where=row_sum > 0,
    )

    mutual_preference = np.sqrt(preference * preference.T)

    nearest = np.argmin(
        np.where(finite_rows, d, np.inf),
        axis=1,
    )
    nearest_valid = np.isfinite(d[np.arange(n), nearest])

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        if nearest_valid[i]:
            j = nearest[i]
            if nearest_valid[j] and nearest[j] == i:
                mutual_nearest[i, j] = 1.0
                mutual_nearest[j, i] = 1.0

    nearest_bonus = 1.0 + 0.35 * mutual_nearest

    sorted_preferences = np.sort(
        np.where(finite_rows, preference, 0.0),
        axis=1,
    )[:, ::-1]

    best = sorted_preferences[:, 0]
    second = (
        sorted_preferences[:, 1]
        if n > 2
        else np.zeros(n, dtype=float)
    )

    regret = np.divide(
        best - second,
        best + eps,
    )
    regret = np.clip(regret, 0.0, 1.0)

    endpoint_regret = 1.0 + 0.25 * (
        regret[:, None] + regret[None, :]
    )

    degree_scale = np.maximum(np.sum(attraction > 0, axis=1), 1.0)
    balance = 1.0 / np.sqrt(degree_scale[:, None] * degree_scale[None, :])
    balance /= np.max(balance) if np.max(balance) > 0 else 1.0

    scores = (
        mutual_preference
        * nearest_bonus
        * endpoint_regret
        * (0.85 + 0.15 * balance)
    )

    scores[~np.isfinite(scores)] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
