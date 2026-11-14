import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distance_matrix)
    valid = finite & off_diagonal

    distances = np.zeros((n, n), dtype=float)
    distances[finite] = np.maximum(distance_matrix[finite], 0.0)

    finite_values = distances[valid]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    if np.all(finite_values == 0.0):
        result = valid.astype(float)
        maximum = float(np.max(result))
        return result / maximum if maximum > 0.0 else result

    eps = np.finfo(float).eps
    positive_values = finite_values[finite_values > 0.0]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, eps)

    safe_distances = np.where(valid, distances, np.inf)

    row_best = np.min(safe_distances, axis=1)
    col_best = np.min(safe_distances, axis=0)

    finite_row = row_best[np.isfinite(row_best)]
    finite_col = col_best[np.isfinite(col_best)]

    row_fallback = float(np.median(finite_row)) if finite_row.size else scale
    col_fallback = float(np.median(finite_col)) if finite_col.size else scale

    row_best = np.where(np.isfinite(row_best), row_best, row_fallback)
    col_best = np.where(np.isfinite(col_best), col_best, col_fallback)

    normalized = np.divide(
        distances,
        scale,
        out=np.zeros_like(distances),
        where=valid,
    )
    distance_score = 1.0 / (1.0 + normalized * normalized)

    row_local = np.divide(
        row_best[:, None],
        row_best[:, None] + distances + 0.5 * scale + eps,
    )
    col_local = np.divide(
        col_best[None, :],
        col_best[None, :] + distances + 0.5 * scale + eps,
    )
    local_score = np.clip(0.5 * (row_local + col_local), 0.0, 1.0)

    base_scores = distance_score * (0.78 + 0.22 * local_score)

    nearest = np.argmin(safe_distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = int(nearest[i])
        if j != i and valid[i, j] and valid[j, i] and nearest[j] == i:
            mutual[i, j] = 1.0

    base_scores *= 1.0 + 0.12 * mutual

    out_degree = np.sum(valid, axis=1)
    in_degree = np.sum(valid, axis=0)
    max_degree = max(n - 1, 1)

    out_bonus = np.clip(
        np.sqrt(max_degree / np.maximum(out_degree, 1)),
        1.0,
        1.25,
    )
    in_bonus = np.clip(
        np.sqrt(max_degree / np.maximum(in_degree, 1)),
        1.0,
        1.25,
    )

    sparsity_factor = (
        1.0
        + 0.04 * (out_bonus[:, None] - 1.0)
        + 0.04 * (in_bonus[None, :] - 1.0)
    )
    base_scores *= sparsity_factor

    attraction = np.zeros((n, n), dtype=float)
    positive_scale = max(scale, eps)
    for factor in (0.7, 1.0, 1.5):
        kernel = np.zeros((n, n), dtype=float)
        kernel[valid] = np.exp(
            -np.sqrt(distances[valid] / (positive_scale * factor))
        )
        attraction += kernel
    attraction /= 3.0

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = np.argsort(distances[i, indices], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.6 * indices.size)
            row_rank[i, indices] = np.exp(-ranks / temperature)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = np.argsort(distances[indices, j], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.6 * indices.size)
            col_rank[indices, j] = np.exp(-ranks / temperature)

    rank_signal = 0.5 * (row_rank + col_rank)

    outgoing_potential = np.zeros(n, dtype=float)
    incoming_potential = np.zeros(n, dtype=float)

    for i in range(n):
        values = attraction[i, valid[i]]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            outgoing_potential[i] = float(np.mean(np.sort(values)[-k:]))

    for j in range(n):
        values = attraction[valid[:, j], j]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            incoming_potential[j] = float(np.mean(np.sort(values)[-k:]))

    future = np.sqrt(
        np.clip(
            incoming_potential[:, None] * outgoing_potential[None, :],
            0.0,
            1.0,
        )
    )

    scores = (
        0.84 * base_scores
        + 0.08 * attraction
        + 0.05 * rank_signal
        + 0.03 * future
    )

    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
