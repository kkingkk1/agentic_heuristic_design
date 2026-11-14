import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distance_matrix) & off_diagonal

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = np.maximum(distance_matrix[valid], 0.0)
    distances[~valid] = np.inf

    finite_values = distances[valid & np.isfinite(distances)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    scale = max(scale, np.finfo(float).eps)

    normalized = np.divide(
        distances,
        scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(distances),
    )

    desirability = np.zeros((n, n), dtype=float)
    finite_normalized = np.isfinite(normalized)
    desirability[finite_normalized] = 1.0 / (
        1.0 + normalized[finite_normalized] ** 2
    )

    row_nearest = np.full(n, scale, dtype=float)
    for i in range(n):
        candidates = distances[i, valid[i]]
        if candidates.size:
            row_nearest[i] = float(np.min(candidates))

    local_ratio = np.divide(
        row_nearest[:, None],
        row_nearest[:, None] + distances + 0.5 * scale,
        out=np.zeros((n, n), dtype=float),
        where=np.isfinite(distances),
    )
    local_ratio = np.clip(local_ratio, 0.0, 1.0)

    scores = desirability * (0.72 + 0.28 * local_ratio)

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = np.argsort(distances[i, indices], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.45 * indices.size)
            row_rank[i, indices] = np.exp(-ranks / temperature)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = np.argsort(distances[indices, j], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.45 * indices.size)
            col_rank[indices, j] = np.exp(-ranks / temperature)

    diversity = 0.78 + 0.11 * row_rank + 0.11 * col_rank
    scores *= diversity

    row_degree = valid.sum(axis=1).astype(float)
    col_degree = valid.sum(axis=0).astype(float)

    outgoing = np.zeros(n, dtype=float)
    incoming = np.zeros(n, dtype=float)

    for i in range(n):
        values = desirability[i, valid[i]]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            outgoing[i] = float(np.mean(np.sort(values)[-k:]))

    for j in range(n):
        values = desirability[valid[:, j], j]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            incoming[j] = float(np.mean(np.sort(values)[-k:]))

    future = (
        0.90
        + 0.05 * outgoing[None, :]
        + 0.05 * incoming[:, None]
    )
    scores *= future

    max_degree = max(n - 1, 1)
    out_sparsity = np.sqrt(max_degree / np.maximum(row_degree, 1.0))
    in_sparsity = np.sqrt(max_degree / np.maximum(col_degree, 1.0))

    out_sparsity = np.clip(out_sparsity, 1.0, 1.35)
    in_sparsity = np.clip(in_sparsity, 1.0, 1.35)

    reachability = (
        1.0
        + 0.05 * (out_sparsity[:, None] - 1.0)
        + 0.05 * (in_sparsity[None, :] - 1.0)
    )
    scores *= reachability

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            nearest[i] = int(indices[np.argmin(distances[i, indices])])

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i and valid[i, j]:
            mutual[i, j] = 1.0

    scores *= 1.0 + 0.18 * mutual
    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
