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

    result = np.zeros((n, n), dtype=float)
    if n == 1:
        return result

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    finite_edges = work[np.isfinite(work)]
    if finite_edges.size == 0:
        return result

    eps = np.finfo(float).eps
    global_scale = max(float(np.median(finite_edges)), eps)

    neighbor_count = min(max(3, int(np.sqrt(n))), n - 1)
    local_scale = np.partition(work, neighbor_count - 1, axis=1)[:, neighbor_count - 1]
    local_scale = np.maximum(local_scale, eps)

    local_ratio = work / local_scale[:, None]
    global_ratio = work / global_scale

    distance_score = (
        0.68 / (1.0 + local_ratio)
        + 0.32 / (1.0 + global_ratio)
    )

    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = 1.0 / (1.0 + ranks)

    local_score = 0.72 * distance_score + 0.28 * rank_score
    local_score[~np.isfinite(local_score)] = 0.0
    np.fill_diagonal(local_score, 0.0)

    reverse_score = local_score.T
    mutual_rank = np.sqrt(np.maximum(local_score, 0.0) *
                          np.maximum(reverse_score, 0.0))

    symmetric_score = 0.5 * (local_score + reverse_score)
    result = 0.58 * mutual_rank + 0.42 * symmetric_score
    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 0.0)

    row_max = result.max(axis=1, keepdims=True)
    result = np.divide(
        result,
        np.maximum(row_max, eps),
        out=np.zeros_like(result),
        where=row_max > 0,
    )

    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = result.max()
    if maximum > 0:
        result /= maximum

    return result
