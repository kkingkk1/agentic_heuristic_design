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

    distances = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=0.0,
    )
    distances = np.maximum(distances, 0.0).astype(float, copy=True)
    np.fill_diagonal(distances, np.inf)

    finite_mask = np.isfinite(distances)
    off_diagonal_mask = ~np.eye(n, dtype=bool)

    finite_off_diagonal = distances[finite_mask & off_diagonal_mask]
    if finite_off_diagonal.size == 0:
        return np.zeros((n, n), dtype=float)

    if np.all(finite_off_diagonal == 0.0):
        scores = np.ones((n, n), dtype=float)
        np.fill_diagonal(scores, 0.0)
        return scores

    positive_distances = finite_off_diagonal[finite_off_diagonal > 0.0]
    if positive_distances.size:
        scale = float(np.median(positive_distances))
    else:
        scale = 1.0
    scale = max(scale, np.finfo(float).eps)

    row_nearest_distance = np.min(distances, axis=1)
    finite_row_nearest = row_nearest_distance[np.isfinite(row_nearest_distance)]
    fallback = float(np.median(finite_row_nearest)) if finite_row_nearest.size else scale
    fallback = max(fallback, scale, np.finfo(float).eps)
    row_nearest_distance = np.where(
        np.isfinite(row_nearest_distance),
        row_nearest_distance,
        fallback,
    )

    normalized = np.divide(
        distances,
        scale,
        out=np.full_like(distances, np.inf),
        where=np.isfinite(distances),
    )
    desirability = 1.0 / (1.0 + normalized * normalized)

    local_ratio = np.divide(
        row_nearest_distance[:, None],
        row_nearest_distance[:, None] + distances + 0.5 * scale,
        out=np.zeros_like(distances),
        where=np.isfinite(distances),
    )
    local_ratio = np.clip(local_ratio, 0.0, 1.0)

    scores = desirability * (0.68 + 0.32 * local_ratio)

    nearest_index = np.argmin(distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest_index[i])
        if j != i and nearest_index[j] == i:
            mutual[i, j] = 1.0

    scores *= 1.0 + 0.30 * mutual

    finite_out_degree = np.sum(finite_mask & off_diagonal_mask, axis=1)
    finite_in_degree = np.sum(finite_mask & off_diagonal_mask, axis=0)

    max_degree = max(n - 1, 1)
    out_sparsity = np.sqrt(max_degree / np.maximum(finite_out_degree, 1))
    in_sparsity = np.sqrt(max_degree / np.maximum(finite_in_degree, 1))

    out_sparsity = np.clip(out_sparsity, 1.0, 1.5)
    in_sparsity = np.clip(in_sparsity, 1.0, 1.5)

    reachability_factor = (
        1.0
        + 0.08 * (out_sparsity[:, None] - 1.0)
        + 0.08 * (in_sparsity[None, :] - 1.0)
    )
    scores *= reachability_factor

    scores[~finite_mask] = 0.0
    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
