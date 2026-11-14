import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distance_matrix)
    np.fill_diagonal(valid, False)

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = np.maximum(distance_matrix[valid], 0.0)
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[valid]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive.size:
        q25, q50, q75 = np.percentile(positive, [25.0, 50.0, 75.0])
        scales = np.array(
            [
                max(float(q25), eps),
                max(float(q50), eps),
                max(float(q75), eps),
            ],
            dtype=float,
        )
    else:
        scales = np.ones(3, dtype=float)

    attraction = np.zeros((n, n), dtype=float)
    for scale in scales:
        values = np.zeros((n, n), dtype=float)
        values[valid] = np.exp(
            -np.sqrt(np.maximum(distances[valid], 0.0) / scale)
        )
        attraction += values
    attraction /= float(scales.size)
    attraction[~valid] = 0.0

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = np.argsort(distances[i, indices], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.55 * indices.size)
            row_rank[i, indices] = np.exp(-ranks / temperature)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = np.argsort(distances[indices, j], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.55 * indices.size)
            col_rank[indices, j] = np.exp(-ranks / temperature)

    row_sum = attraction.sum(axis=1)
    col_sum = attraction.sum(axis=0)

    row_entropy = np.zeros(n, dtype=float)
    col_entropy = np.zeros(n, dtype=float)

    for i in range(n):
        values = attraction[i, valid[i]]
        total = float(values.sum())
        if values.size > 1 and total > eps:
            probabilities = values / total
            entropy = -float(np.sum(probabilities * np.log(probabilities + eps)))
            row_entropy[i] = entropy / np.log(float(values.size))

    for j in range(n):
        values = attraction[valid[:, j], j]
        total = float(values.sum())
        if values.size > 1 and total > eps:
            probabilities = values / total
            entropy = -float(np.sum(probabilities * np.log(probabilities + eps)))
            col_entropy[j] = entropy / np.log(float(values.size))

    outgoing_potential = np.zeros(n, dtype=float)
    incoming_potential = np.zeros(n, dtype=float)

    for j in range(n):
        values = attraction[j, valid[j]]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            top_values = np.sort(values)[-k:]
            outgoing_potential[j] = float(np.mean(top_values))

    for i in range(n):
        values = attraction[valid[:, i], i]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            top_values = np.sort(values)[-k:]
            incoming_potential[i] = float(np.mean(top_values))

    future = np.sqrt(
        np.clip(
            outgoing_potential[None, :] * incoming_potential[:, None],
            0.0,
            1.0,
        )
    )

    row_competition = row_sum / max(float(np.max(row_sum)), eps)
    col_competition = col_sum / max(float(np.max(col_sum)), eps)

    balance = 1.0 / np.sqrt(
        (0.55 + row_competition[:, None])
        * (0.55 + col_competition[None, :])
    )
    balance = np.clip(balance, 0.0, 1.0)

    diversity = 0.5 * (
        row_entropy[:, None] + col_entropy[None, :]
    )

    rank_signal = 0.5 * (row_rank + col_rank)

    scores = (
        0.42 * attraction
        + 0.18 * rank_signal
        + 0.22 * future
        + 0.10 * balance
        + 0.08 * diversity
    )

    scores *= 0.78 + 0.22 * np.sqrt(np.clip(balance, 0.0, 1.0))
    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
