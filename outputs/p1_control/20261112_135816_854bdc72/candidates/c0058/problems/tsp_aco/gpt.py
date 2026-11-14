import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distances must be non-negative")

    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_edges = d[np.isfinite(d)]
    positive_edges = finite_edges[finite_edges > 0]

    if positive_edges.size:
        scale_floor = max(
            np.finfo(float).eps,
            float(np.median(positive_edges)) * 1e-12,
        )
    else:
        scale_floor = 1e-12

    positive_d = np.where(np.isfinite(d), np.maximum(d, scale_floor), np.inf)

    order = np.sort(positive_d, axis=1)
    k_small = min(max(2, int(np.sqrt(n))), n - 1)
    local_scale = np.mean(order[:, :k_small], axis=1)
    local_scale = np.maximum(local_scale, scale_floor)

    endpoint_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_cost = positive_d / np.maximum(endpoint_scale, scale_floor)

    finite_relative = relative_cost[np.isfinite(relative_cost)]
    temperature = (
        float(np.median(finite_relative))
        if finite_relative.size
        else 1.0
    )
    temperature = max(temperature, 1e-6)

    closeness = np.exp(
        -np.minimum(relative_cost / temperature, 60.0)
    )
    closeness[~np.isfinite(closeness)] = 0.0

    inverse = 1.0 / positive_d
    inverse[~np.isfinite(inverse)] = 0.0

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max = np.maximum(row_max, scale_floor)
    normalized_inverse = inverse / row_max

    row_sum = np.sum(normalized_inverse, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, scale_floor)
    row_competition = normalized_inverse / row_sum

    symmetric_competition = np.sqrt(
        row_competition * row_competition.T
    )

    nearest = np.argmin(positive_d, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest[i])
        if j != i and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    row_sorted = order[:, : min(3, n - 1)]
    if row_sorted.shape[1] >= 2:
        regret = row_sorted[:, 1] - row_sorted[:, 0]
        regret = regret / np.maximum(row_sorted[:, 1], scale_floor)
    else:
        regret = np.zeros(n, dtype=float)

    regret_scale = np.sqrt(
        np.maximum(regret[:, None], 0.0)
        * np.maximum(regret[None, :], 0.0)
    )
    regret_bonus = np.clip(regret_scale, 0.0, 1.0)

    score = (
        0.34 * closeness
        + 0.38 * symmetric_competition
        + 0.18 * np.sqrt(
            np.maximum(row_competition * row_competition.T, 0.0)
        )
        + 0.06 * mutual_nearest
        + 0.04 * regret_bonus * closeness
    )

    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)
    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score
