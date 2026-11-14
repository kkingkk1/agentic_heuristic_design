import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(d) & ~np.eye(n, dtype=bool) & (d >= 0)
    if not np.any(valid):
        return result

    positive = d[valid & (d > 0)]
    epsilon = np.finfo(float).eps
    replacement = float(np.median(positive)) if positive.size else 1.0

    safe = np.where(valid, np.maximum(d, epsilon), replacement)
    safe[~valid] = np.inf

    ranks = np.zeros((n, n), dtype=float)
    for i in range(n):
        row_valid = valid[i]
        count = int(np.sum(row_valid))
        if count == 0:
            continue

        order = np.argsort(np.where(row_valid, safe[i], np.inf), kind="mergesort")
        rank = 0
        for j in order:
            if row_valid[j]:
                ranks[i, j] = rank / max(count - 1, 1)
                rank += 1

    directed_kernel = np.zeros((n, n), dtype=float)
    directed_kernel[valid] = np.exp(-3.0 * ranks[valid])

    kernel = 0.5 * (directed_kernel + directed_kernel.T)
    kernel[~valid] = 0.0
    np.fill_diagonal(kernel, 0.0)

    degree = np.sum(kernel, axis=1)
    degree_safe = np.maximum(degree, epsilon)

    normalized_kernel = kernel / np.sqrt(
        degree_safe[:, None] * degree_safe[None, :]
    )

    shared_neighbors = normalized_kernel @ normalized_kernel
    shared_neighbors[~valid] = 0.0
    np.fill_diagonal(shared_neighbors, 0.0)

    three_step = normalized_kernel @ normalized_kernel @ normalized_kernel
    three_step[~valid] = 0.0
    np.fill_diagonal(three_step, 0.0)

    def normalize(values: np.ndarray) -> np.ndarray:
        maximum = float(np.max(values))
        if maximum <= epsilon:
            return np.zeros_like(values)
        return values / maximum

    direct = normalize(kernel)
    local_support = normalize(shared_neighbors)
    cycle_support = normalize(three_step)

    score = (
        0.35 * direct
        + 0.30 * local_support
        + 0.35 * cycle_support
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > epsilon:
        score /= maximum

    return score.astype(float, copy=False)
