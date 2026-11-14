import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distance_matrix)
    np.fill_diagonal(valid, False)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = np.maximum(distance_matrix[valid], 0.0)
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[valid]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        scale = max(float(np.median(positive_values)), eps)
    else:
        scores = np.zeros((n, n), dtype=float)
        scores[valid] = 1.0
        return scores

    normalized = np.divide(
        distances,
        scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(distances),
    )

    # Preserve strong distance discrimination while avoiding an overly sharp penalty.
    desirability = np.zeros((n, n), dtype=float)
    finite_normalized = np.isfinite(normalized)
    desirability[finite_normalized] = 1.0 / (
        1.0 + np.square(np.maximum(normalized[finite_normalized], 0.0))
    )

    row_nearest = np.full(n, scale, dtype=float)
    for i in range(n):
        candidates = distances[i, valid[i]]
        if candidates.size:
            row_nearest[i] = float(np.min(candidates))

    local_ratio = np.zeros((n, n), dtype=float)
    denominator = row_nearest[:, None] + distances + 0.5 * scale
    usable = valid & np.isfinite(denominator)
    local_ratio[usable] = np.clip(
        row_nearest[:, None][usable] / denominator[usable],
        0.0,
        1.0,
    )

    scores = desirability * (0.70 + 0.30 * local_ratio)

    # Directional rank signals from both endpoints, kept deliberately restrained.
    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = np.argsort(distances[i, indices], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.38 * indices.size)
            row_rank[i, indices] = np.exp(-ranks / temperature)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = np.argsort(distances[indices, j], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.38 * indices.size)
            col_rank[indices, j] = np.exp(-ranks / temperature)

    directional_factor = 0.76 + 0.12 * row_rank + 0.12 * col_rank

    # Bounded continuation signal, inspired by A's two-step potential.
    attraction = desirability
    outgoing_potential = np.zeros(n, dtype=float)
    incoming_potential = np.zeros(n, dtype=float)

    for i in range(n):
        values = attraction[i, valid[i]]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            outgoing_potential[i] = float(
                np.mean(np.sort(values)[-k:])
            )

    for j in range(n):
        values = attraction[valid[:, j], j]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            incoming_potential[j] = float(
                np.mean(np.sort(values)[-k:])
            )

    continuation = (
        0.88
        + 0.06 * outgoing_potential[:, None]
        + 0.06 * incoming_potential[None, :]
    )

    # Mildly favor edges that preserve finite outgoing and incoming options.
    out_degree = valid.sum(axis=1).astype(float)
    in_degree = valid.sum(axis=0).astype(float)
    degree_scale = max(float(n - 1), 1.0)

    out_factor = np.clip(
        np.sqrt(degree_scale / np.maximum(out_degree, 1.0)),
        1.0,
        1.5,
    )
    in_factor = np.clip(
        np.sqrt(degree_scale / np.maximum(in_degree, 1.0)),
        1.0,
        1.5,
    )

    reachability = (
        1.0
        + 0.06 * (out_factor[:, None] - 1.0)
        + 0.06 * (in_factor[None, :] - 1.0)
    )

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

    scores *= directional_factor
    scores *= continuation
    scores *= reachability
    scores *= 1.0 + 0.20 * mutual

    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
