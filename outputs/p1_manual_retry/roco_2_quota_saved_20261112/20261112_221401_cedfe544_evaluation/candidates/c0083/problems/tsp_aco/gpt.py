import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    raw = np.where(np.isfinite(distance_matrix), distance_matrix, np.inf)
    raw = np.maximum(raw, 0.0)
    np.fill_diagonal(raw, np.inf)

    finite_values = raw[np.isfinite(raw)]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif finite_values.size:
        scale = 1.0
    else:
        return np.zeros((n, n), dtype=float)

    scale = max(scale, np.finfo(float).eps)

    # Preserve the raw-distance ordering as the dominant signal.
    normalized = raw / scale
    desirability = 1.0 / (1.0 + normalized)
    desirability[~np.isfinite(desirability)] = 0.0

    # Use symmetric distances only for local structural signals.
    symmetric = 0.5 * (raw + raw.T)
    np.fill_diagonal(symmetric, np.inf)

    local_rank = np.zeros((n, n), dtype=float)
    for i in range(n):
        finite = np.isfinite(symmetric[i])
        count = int(np.count_nonzero(finite))
        if count == 0:
            continue

        order = np.argsort(symmetric[i], kind="mergesort")
        order = order[finite[order]]

        if count == 1:
            local_rank[i, order[0]] = 1.0
        else:
            ranks = np.arange(count, dtype=float)
            local_rank[i, order] = 1.0 - ranks / (count - 1.0)

    # Mutual nearest neighbors provide a small, explicit local-structure bonus.
    nearest = np.argmin(symmetric, axis=1)
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = int(nearest[i])
        if j != i and nearest[j] == i:
            mutual[i, j] = 1.0

    # Symmetric local rank is restrained so distance remains the main factor.
    local_factor = 0.80 + 0.20 * np.sqrt(local_rank * local_rank.T)
    mutual_factor = 1.0 + 0.25 * mutual

    scores = desirability * local_factor * mutual_factor
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
