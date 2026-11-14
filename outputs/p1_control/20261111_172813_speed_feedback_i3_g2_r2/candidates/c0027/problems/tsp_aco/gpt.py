import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return robust bilateral edge-priority indicators for stochastic TSP sampling.
    Larger values indicate edges that are unusually short for both endpoints.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n == 0:
        return result

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & ~diagonal

    if not np.any(valid):
        return result

    positive = d[valid & (d > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    floor = max(global_scale * 1e-12, np.finfo(float).eps)

    row_center = np.full(n, global_scale, dtype=float)
    col_center = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        if values.size:
            row_center[i] = float(np.median(values))

    for j in range(n):
        values = d[valid[:, j], j]
        if values.size:
            col_center[j] = float(np.median(values))

    row_scale = np.full(n, global_scale, dtype=float)
    col_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        if values.size:
            deviation = np.median(np.abs(values - row_center[i]))
            row_scale[i] = max(float(deviation), floor)

    for j in range(n):
        values = d[valid[:, j], j]
        if values.size:
            deviation = np.median(np.abs(values - col_center[j]))
            col_scale[j] = max(float(deviation), floor)

    row_advantage = np.zeros((n, n), dtype=float)
    col_advantage = np.zeros((n, n), dtype=float)

    row_advantage[valid] = (
        row_center[:, None][valid] - d[valid]
    ) / row_scale[:, None][valid]

    col_advantage[valid] = (
        col_center[None, :][valid] - d[valid]
    ) / col_scale[None, :][valid]

    row_advantage = np.maximum(row_advantage, 0.0)
    col_advantage = np.maximum(col_advantage, 0.0)

    agreement = np.sqrt(row_advantage * col_advantage)
    support = np.minimum(row_advantage, col_advantage)

    endpoint_scale = np.sqrt(
        np.maximum(row_center[:, None], floor)
        * np.maximum(col_center[None, :], floor)
    )
    accessibility = np.zeros((n, n), dtype=float)
    accessibility[valid] = 1.0 / np.sqrt(
        1.0 + np.maximum(d[valid], 0.0) / endpoint_scale[valid]
    )

    score = (0.65 * agreement + 0.25 * support + 0.10 * accessibility)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
