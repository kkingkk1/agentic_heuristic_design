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

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = np.maximum(distance_matrix[valid], 0.0)
    np.fill_diagonal(distances, np.inf)

    finite = distances[valid]
    if finite.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = finite[finite > 0.0]

    if positive.size:
        scales = np.percentile(positive, [25.0, 50.0, 75.0])
        scales = np.maximum(scales.astype(float), eps)
    else:
        scales = np.ones(3, dtype=float)

    attraction = np.zeros((n, n), dtype=float)
    for scale in scales:
        attraction[valid] += np.exp(
            -np.sqrt(np.maximum(distances[valid], 0.0) / scale)
        )
    attraction /= float(scales.size)
    attraction[~valid] = 0.0

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    row_degree = valid.sum(axis=1).astype(float)
    col_degree = valid.sum(axis=0).astype(float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size:
            order = np.argsort(distances[i, indices], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.60 * np.sqrt(indices.size))
            row_rank[i, indices] = np.exp(-ranks / temperature)

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size:
            order = np.argsort(distances[indices, j], kind="stable")
            ranks = np.empty(indices.size, dtype=float)
            ranks[order] = np.arange(indices.size, dtype=float)
            temperature = max(1.0, 0.60 * np.sqrt(indices.size))
            col_rank[indices, j] = np.exp(-ranks / temperature)

    row_sum = attraction.sum(axis=1)
    col_sum = attraction.sum(axis=0)

    row_prob = np.divide(
        attraction,
        row_sum[:, None],
        out=np.zeros_like(attraction),
        where=row_sum[:, None] > eps,
    )
    col_prob = np.divide(
        attraction,
        col_sum[None, :],
        out=np.zeros_like(attraction),
        where=col_sum[None, :] > eps,
    )

    row_entropy = np.zeros(n, dtype=float)
    col_entropy = np.zeros(n, dtype=float)

    for i in range(n):
        count = int(row_degree[i])
        if count > 1:
            p = row_prob[i, valid[i]]
            entropy = -np.sum(p * np.log(np.maximum(p, eps)))
            row_entropy[i] = entropy / np.log(float(count))

    for j in range(n):
        count = int(col_degree[j])
        if count > 1:
            p = col_prob[valid[:, j], j]
            entropy = -np.sum(p * np.log(np.maximum(p, eps)))
            col_entropy[j] = entropy / np.log(float(count))

    row_best = np.zeros(n, dtype=float)
    col_best = np.zeros(n, dtype=float)

    for i in range(n):
        values = attraction[i, valid[i]]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            row_best[i] = np.mean(np.sort(values)[-k:])

    for j in range(n):
        values = attraction[valid[:, j], j]
        if values.size:
            k = max(1, int(np.ceil(np.sqrt(values.size))))
            col_best[j] = np.mean(np.sort(values)[-k:])

    outgoing = row_best[:, None]
    incoming = col_best[None, :]

    forward_potential = np.sqrt(
        np.clip(outgoing * incoming, 0.0, 1.0)
    )

    continuation = np.zeros((n, n), dtype=float)
    for i in range(n):
        targets = np.flatnonzero(valid[i])
        for j in targets:
            next_values = attraction[j, valid[j]]
            if next_values.size:
                k = max(1, int(np.ceil(np.sqrt(next_values.size))))
                continuation[i, j] = np.mean(np.sort(next_values)[-k:])

    predecessor = np.zeros((n, n), dtype=float)
    for i in range(n):
        sources = np.flatnonzero(valid[:, i])
        for j in sources:
            previous_values = attraction[valid[:, j], j]
            if previous_values.size:
                k = max(1, int(np.ceil(np.sqrt(previous_values.size))))
                predecessor[j, i] = np.mean(
                    np.sort(previous_values)[-k:]
                )

    long_term = np.sqrt(
        np.clip(continuation * predecessor, 0.0, 1.0)
    )

    row_max = np.max(attraction, axis=1, keepdims=True)
    col_max = np.max(attraction, axis=0, keepdims=True)

    row_margin = np.divide(
        attraction,
        row_max,
        out=np.zeros_like(attraction),
        where=row_max > eps,
    )
    col_margin = np.divide(
        attraction,
        col_max,
        out=np.zeros_like(attraction),
        where=col_max > eps,
    )

    mutual = np.sqrt(np.clip(row_margin * col_margin, 0.0, 1.0))

    row_coverage = np.sqrt(
        np.clip(row_entropy[:, None] * row_prob, 0.0, 1.0)
    )
    col_coverage = np.sqrt(
        np.clip(col_entropy[None, :] * col_prob, 0.0, 1.0)
    )

    edge_diversity = np.sqrt(
        np.clip(row_coverage * col_coverage, 0.0, 1.0)
    )

    row_competition = row_sum / max(float(np.max(row_sum)), eps)
    col_competition = col_sum / max(float(np.max(col_sum)), eps)

    scarcity = 1.0 / np.sqrt(
        (0.65 + row_competition[:, None])
        * (0.65 + col_competition[None, :])
    )
    scarcity = np.clip(scarcity, 0.0, 1.0)

    degree_factor = np.sqrt(
        np.minimum(
            row_degree[:, None] / max(float(n - 1), 1.0),
            col_degree[None, :] / max(float(n - 1), 1.0),
        )
    )
    mutual_bonus = mutual * (0.55 + 0.45 * degree_factor)

    rank_signal = 0.5 * (row_rank + col_rank)

    scores = (
        0.40 * attraction
        + 0.15 * rank_signal
        + 0.19 * long_term
        + 0.10 * forward_potential
        + 0.08 * edge_diversity
        + 0.05 * scarcity
        + 0.03 * mutual_bonus
    )

    scores *= 0.82 + 0.18 * np.sqrt(
        np.clip(edge_diversity * scarcity, 0.0, 1.0)
    )

    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return scores.astype(float, copy=False)
