import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    if np.any(distances < 0):
        raise ValueError("distance_matrix must contain non-negative distances")

    result = np.zeros_like(distances, dtype=float)
    if n == 1:
        return result

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    row_scale = np.partition(work, min(n - 2, 3), axis=1)[:, min(n - 2, 3)]
    row_scale = np.maximum(row_scale, np.finfo(float).eps)

    relative_distance = work / row_scale[:, None]
    inverse_attractiveness = 1.0 / (1.0 + relative_distance)

    order = np.argsort(work, axis=1)
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_attractiveness = 1.0 / (1.0 + ranks)

    local_score = 0.65 * inverse_attractiveness + 0.35 * rank_attractiveness
    local_score[~np.isfinite(local_score)] = 0.0
    np.fill_diagonal(local_score, 0.0)

    symmetric_score = 0.5 * (local_score + local_score.T)
    mutual_score = np.sqrt(np.maximum(local_score, 0.0) *
                           np.maximum(local_score.T, 0.0))

    result = 0.55 * symmetric_score + 0.45 * mutual_score
    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = result.max()
    if maximum > 0:
        result /= maximum

    return result
