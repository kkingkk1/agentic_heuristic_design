import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distance_matrix)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = np.maximum(distance_matrix[valid], 0.0)
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[valid & np.isfinite(distances)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = finite_values[finite_values > 0.0]
    if positive.size:
        q20, q50, q80 = np.percentile(positive, [20.0, 50.0, 80.0])
        low = max(float(q20), np.finfo(float).eps)
        scale = max(float(q50), np.finfo(float).eps)
        high = max(float(q80), low)
    else:
        low = scale = high = 1.0

    spread = high / low
    decay = float(np.clip(0.82 + 0.10 * np.log1p(spread), 0.78, 1.30))
    exponent = float(np.clip(1.08 - 0.07 * np.log1p(spread), 0.82, 1.20))

    normalized = np.divide(
        distances,
        scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(distances),
    )

    attraction = np.zeros((n, n), dtype=float)
    finite_norm = np.isfinite(normalized)
    attraction[finite_norm] = 1.0 / (
        1.0
        + decay * np.power(
            np.maximum(normalized[finite_norm], 0.0),
            exponent,
        )
    )

    row_counts = valid.sum(axis=1).astype(float)
    col_counts = valid.sum(axis=0).astype(float)

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

    outgoing_potential = np.zeros(n, dtype=float)
    incoming_potential = np.zeros(n, dtype=float)

    for j in range(n):
        values = attraction[j, valid[j]]
        if values.size:
            sorted_values = np.sort(values)[::-1]
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            outgoing_potential[j] = float(np.mean(sorted_values[:k]))

    for i in range(n):
        values = attraction[valid[:, i], i]
        if values.size:
            sorted_values = np.sort(values)[::-1]
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            incoming_potential[i] = float(np.mean(sorted_values[:k]))

    future = (
        0.58
        + 0.21 * outgoing_potential[None, :]
        + 0.21 * incoming_potential[:, None]
    )

    degree_scale = max(1.0, float(n - 1))
    reachability = np.sqrt(
        np.clip(
            (row_counts[:, None] + 1.0)
            * (col_counts[None, :] + 1.0)
            / ((degree_scale + 1.0) ** 2),
            0.0,
            1.0,
        )
    )
    reachability = 0.82 + 0.18 * reachability

    diversity = 0.50 + 0.25 * row_rank + 0.25 * col_rank

    nearest = np.full(n, -1, dtype=int)
    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            nearest[i] = int(indices[np.argmin(distances[i, indices])])

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest[i]
        if j >= 0 and nearest[j] == i and valid[j, i]:
            mutual[i, j] = 1.0

    scores = (
        attraction
        * diversity
        * future
        * reachability
        * (1.0 + 0.30 * mutual)
    )

    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
