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

    finite_edges = work[np.isfinite(work)]
    positive_edges = finite_edges[finite_edges > 0]

    if positive_edges.size:
        global_scale = float(np.median(positive_edges))
    elif finite_edges.size:
        global_scale = 1.0
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    neighbor_count = min(max(2, n // 5), 8, n - 1)
    local_scale = np.partition(
        work, neighbor_count - 1, axis=1
    )[:, neighbor_count - 1]
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    row_relative = work / local_scale[:, None]
    col_relative = work / local_scale[None, :]

    row_attraction = 1.0 / (1.0 + row_relative)
    col_attraction = 1.0 / (1.0 + col_relative)

    row_rank_order = np.argsort(work, axis=1, kind="stable")
    row_ranks = np.empty_like(work, dtype=float)
    row_ranks[
        np.arange(n)[:, None],
        row_rank_order
    ] = np.arange(n, dtype=float)

    col_rank_order = np.argsort(work, axis=0, kind="stable")
    col_ranks = np.empty_like(work, dtype=float)
    col_ranks[
        col_rank_order,
        np.arange(n)[None, :]
    ] = np.arange(n, dtype=float)

    row_rank_score = 1.0 / (1.0 + row_ranks)
    col_rank_score = 1.0 / (1.0 + col_ranks)

    row_score = 0.72 * row_attraction + 0.28 * row_rank_score
    col_score = 0.72 * col_attraction + 0.28 * col_rank_score

    mutual_score = np.sqrt(
        np.maximum(row_score, 0.0) * np.maximum(col_score, 0.0)
    )
    balanced_score = 0.5 * (row_score + col_score)

    global_quality = 1.0 / (1.0 + work / global_scale)

    result = (
        0.45 * mutual_score
        + 0.30 * balanced_score
        + 0.15 * global_quality
        + 0.10 * np.sqrt(
            np.maximum(row_rank_score, 0.0)
            * np.maximum(col_rank_score, 0.0)
        )
    )

    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if maximum > 0.0:
        result /= maximum

    return result
