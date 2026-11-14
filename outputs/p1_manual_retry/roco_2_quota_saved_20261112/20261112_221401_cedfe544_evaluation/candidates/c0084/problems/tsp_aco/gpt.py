import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    raw = np.where(np.isfinite(distance_matrix), distance_matrix, np.inf)
    raw = np.maximum(raw, 0.0)
    np.fill_diagonal(raw, np.inf)

    distances = (raw + raw.T) / 2.0
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[np.isfinite(distances)]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif finite_values.size:
        scale = 1.0
    else:
        return np.zeros((n, n), dtype=float)

    scale = max(scale, np.finfo(float).eps)

    normalized = distances / scale
    distance_score = 1.0 / (1.0 + normalized * normalized)

    nearest = np.min(distances, axis=1)
    finite_nearest = np.isfinite(nearest)
    nearest_safe = np.where(finite_nearest, nearest, scale)

    rank_score = np.zeros((n, n), dtype=float)
    mutual = np.zeros((n, n), dtype=float)

    nearest_indices = np.argmin(distances, axis=1)

    for i in range(n):
        valid = np.isfinite(distances[i])
        count = int(np.count_nonzero(valid))
        if count == 0:
            continue

        order = np.argsort(np.where(valid, distances[i], np.inf), kind="stable")
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n, dtype=float)

        rank_score[i, valid] = 1.0 - rank[valid] / max(count - 1, 1)

        j = int(nearest_indices[i])
        if valid[j] and nearest_indices[j] == i:
            mutual[i, j] = 1.0

    symmetric_rank = 0.5 * (rank_score + rank_score.T)

    endpoint_scale = np.sqrt(np.outer(nearest_safe, nearest_safe))
    scarcity = endpoint_scale / (distances + endpoint_scale)
    scarcity = np.clip(scarcity, 0.0, 1.0)

    scores = distance_score
    scores *= 0.72 + 0.18 * symmetric_rank
    scores *= 0.78 + 0.22 * scarcity
    scores *= 1.0 + 0.30 * mutual

    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(scores.max())
    if maximum > 0.0:
        scores /= maximum

    return scores
