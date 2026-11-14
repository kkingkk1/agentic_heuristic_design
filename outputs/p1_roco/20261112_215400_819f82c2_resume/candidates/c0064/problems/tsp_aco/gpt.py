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

    finite_offdiag = np.isfinite(d)
    np.fill_diagonal(finite_offdiag, False)

    valid_values = d[finite_offdiag]
    if valid_values.size == 0:
        return scores

    positive_values = valid_values[valid_values > 0]
    fallback_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )

    d[~np.isfinite(d)] = np.inf
    d[d < 0] = 0.0
    np.fill_diagonal(d, np.inf)

    nearest = np.min(d, axis=1)
    nearest[~np.isfinite(nearest)] = fallback_scale

    finite_d = np.where(np.isfinite(d), d, np.nan)
    row_median = np.nanmedian(finite_d, axis=1)
    row_median[~np.isfinite(row_median)] = fallback_scale

    local_scale = (
        0.65 * nearest
        + 0.35 * row_median
    )
    local_scale[~np.isfinite(local_scale)] = fallback_scale
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    preference = np.zeros((n, n), dtype=float)

    decay_power = 0.9
    for i in range(n):
        valid = np.isfinite(d[i])
        if not np.any(valid):
            continue

        scaled_distance = d[i, valid] / local_scale[i]
        preference[i, valid] = np.exp(
            -np.power(scaled_distance, decay_power)
        )

        row_sum = preference[i].sum()
        if row_sum > 0:
            preference[i] /= row_sum

    bidirectional = np.sqrt(preference * preference.T)

    nearest_neighbor = np.zeros((n, n), dtype=float)
    for i in range(n):
        finite_row = np.where(np.isfinite(d[i]), d[i], np.inf)
        j = int(np.argmin(finite_row))

        if np.isfinite(finite_row[j]):
            nearest_neighbor[i, j] = 1.0

    mutual_bonus = 1.0 + 0.7 * (
        nearest_neighbor * nearest_neighbor.T
    )

    scores = bidirectional * mutual_bonus
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0:
        scores /= maximum

    return scores
