import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & off_diagonal

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    safe = np.where(finite, np.maximum(distances, 0.0), np.inf)
    values = safe[finite]

    positive = values[values > 0]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    row_scales = np.full(n, global_scale, dtype=float)
    for i in range(n):
        row_values = safe[i, finite[i]]
        positive_row = row_values[row_values > 0]
        if positive_row.size:
            row_scales[i] = max(
                float(np.median(positive_row)),
                np.finfo(float).eps,
            )

    normalized = safe / row_scales[:, None]
    valid = np.isfinite(normalized)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (1.0 + normalized[valid])

    proximity = np.zeros((n, n), dtype=float)
    proximity[valid] = np.exp(-normalized[valid])

    ranks = np.zeros((n, n), dtype=float)
    nearest = np.full(n, -1, dtype=int)
    second_nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[np.argsort(safe[i, candidates], kind="stable")]
        ranks[i, order] = 1.0 / (1.0 + np.arange(order.size))

        nearest[i] = int(order[0])
        if order.size > 1:
            second_nearest[i] = int(order[1])

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i:
            mutual_nearest[i, j] = 1.0

    candidate_bonus = np.zeros((n, n), dtype=float)
    for i in range(n):
        if nearest[i] >= 0:
            candidate_bonus[i, nearest[i]] = 1.0
        if second_nearest[i] >= 0:
            candidate_bonus[i, second_nearest[i]] = 0.5

    row_score = (
        0.45 * inverse
        + 0.25 * proximity
        + 0.20 * ranks
        + 0.07 * candidate_bonus
        + 0.03 * mutual_nearest
    )

    score = 0.5 * (row_score + row_score.T)
    score[~(valid | valid.T)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return np.maximum(score, 0.0)
