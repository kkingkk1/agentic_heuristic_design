import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = np.array(distance_matrix, dtype=float, copy=True)
    d[~np.isfinite(d)] = np.inf
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros_like(d)

    positive_values = finite_values[finite_values > 0]
    scale = np.median(positive_values) if positive_values.size else 1.0
    scale = max(float(scale), np.finfo(float).eps)

    safe_d = np.where(np.isfinite(d), np.maximum(d, 0.0), np.inf)

    nearest = np.min(safe_d, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, scale)

    k = max(1, min(5, n - 1))
    local_sorted = np.sort(safe_d, axis=1)
    local_neighbors = local_sorted[:, :k]
    local_neighbors = np.where(np.isfinite(local_neighbors), local_neighbors, scale)
    local_scale = np.mean(local_neighbors, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    distance_advantage = np.zeros_like(safe_d)
    finite_mask = np.isfinite(safe_d)
    distance_advantage[finite_mask] = np.exp(
        -safe_d[finite_mask] / (scale + 0.5 * local_scale[:, None][finite_mask])
    )

    relative_distance = np.zeros_like(safe_d)
    relative_distance[finite_mask] = (
        local_scale[:, None][finite_mask]
        / (safe_d[finite_mask] + local_scale[:, None][finite_mask])
    )

    row_rank = np.zeros_like(safe_d)
    for i in range(n):
        order = np.argsort(safe_d[i])
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(n, dtype=float)
        row_rank[i] = np.exp(-rank / max(1.0, float(k)))

    mutual_nearest = np.zeros_like(safe_d)
    nearest_index = np.argmin(safe_d, axis=1)
    for i in range(n):
        j = nearest_index[i]
        if np.isfinite(safe_d[i, j]) and nearest_index[j] == i:
            mutual_nearest[i, j] = 1.0

    scarcity_i = local_scale / (np.mean(local_scale) + np.finfo(float).eps)
    scarcity_j = scarcity_i[None, :]
    scarcity = np.sqrt(scarcity_i[:, None] * scarcity_j)

    score = (
        0.40 * distance_advantage
        + 0.25 * relative_distance
        + 0.20 * row_rank
        + 0.10 * scarcity
        + 0.05 * mutual_nearest
    )

    score = 0.5 * (score + score.T)
    score[~finite_mask] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
