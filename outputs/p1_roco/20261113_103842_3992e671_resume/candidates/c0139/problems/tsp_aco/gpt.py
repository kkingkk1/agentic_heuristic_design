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

    eps = np.finfo(float).eps
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(distance_matrix) & off_diagonal

    distances = np.zeros((n, n), dtype=float)
    distances[valid] = np.maximum(distance_matrix[valid], 0.0)

    values = distances[valid]
    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive = values[values > 0.0]
    if positive.size:
        scales = np.maximum(
            np.percentile(positive, [25.0, 50.0, 75.0]).astype(float),
            eps,
        )
        distance_scale = max(float(np.median(positive)), eps)
    else:
        scales = np.ones(3, dtype=float)
        distance_scale = 1.0

    attraction = np.zeros((n, n), dtype=float)
    for scale in scales:
        attraction[valid] += np.exp(
            -np.sqrt(distances[valid] / scale)
        )
    attraction /= float(scales.size)
    attraction[~valid] = 0.0

    normalized_distance = np.zeros_like(distances)
    normalized_distance[valid] = distances[valid] / distance_scale
    distance_score = np.zeros_like(distances)
    distance_score[valid] = 1.0 / (
        1.0 + np.power(normalized_distance[valid], 1.5)
    )

    safe_distances = np.where(valid, distances, np.inf)
    row_best = np.min(safe_distances, axis=1)
    col_best = np.min(safe_distances, axis=0)

    finite_row_best = row_best[np.isfinite(row_best)]
    finite_col_best = col_best[np.isfinite(col_best)]
    row_fallback = (
        float(np.median(finite_row_best))
        if finite_row_best.size
        else distance_scale
    )
    col_fallback = (
        float(np.median(finite_col_best))
        if finite_col_best.size
        else distance_scale
    )
    row_best = np.where(np.isfinite(row_best), row_best, row_fallback)
    col_best = np.where(np.isfinite(col_best), col_best, col_fallback)

    row_gap = np.maximum(distances - row_best[:, None], 0.0)
    col_gap = np.maximum(distances - col_best[None, :], 0.0)

    row_local = 1.0 / (
        1.0 + row_gap / (distance_scale + row_best[:, None] + eps)
    )
    col_local = 1.0 / (
        1.0 + col_gap / (distance_scale + col_best[None, :] + eps)
    )
    local_score = np.clip(0.5 * (row_local + col_local), 0.0, 1.0)
    local_score[~valid] = 0.0

    row_rank = np.zeros_like(distances)
    col_rank = np.zeros_like(distances)

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

    rank_signal = 0.5 * (row_rank + col_rank)

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
    row_degree = valid.sum(axis=1)
    col_degree = valid.sum(axis=0)

    for i in range(n):
        count = int(row_degree[i])
        if count > 1:
            probability = row_prob[i, valid[i]]
            row_entropy[i] = (
                -np.sum(probability * np.log(np.maximum(probability, eps)))
                / np.log(float(count))
            )

    for j in range(n):
        count = int(col_degree[j])
        if count > 1:
            probability = col_prob[valid[:, j], j]
            col_entropy[j] = (
                -np.sum(probability * np.log(np.maximum(probability, eps)))
                / np.log(float(count))
            )

    row_coverage = np.sqrt(
        np.clip(row_entropy[:, None] * row_prob, 0.0, 1.0)
    )
    col_coverage = np.sqrt(
        np.clip(col_entropy[None, :] * col_prob, 0.0, 1.0)
    )
    edge_diversity = np.sqrt(
        np.clip(row_coverage * col_coverage, 0.0, 1.0)
    )

    row_best_attraction = np.zeros(n, dtype=float)
    col_best_attraction = np.zeros(n, dtype=float)

    for i in range(n):
        candidate_values = attraction[i, valid[i]]
        if candidate_values.size:
            k = max(1, int(np.ceil(np.sqrt(candidate_values.size))))
            row_best_attraction[i] = np.mean(
                np.sort(candidate_values)[-k:]
            )

    for j in range(n):
        candidate_values = attraction[valid[:, j], j]
        if candidate_values.size:
            k = max(1, int(np.ceil(np.sqrt(candidate_values.size))))
            col_best_attraction[j] = np.mean(
                np.sort(candidate_values)[-k:]
            )

    continuation = np.zeros_like(attraction)
    predecessor = np.zeros_like(attraction)

    for i in range(n):
        targets = np.flatnonzero(valid[i])
        for j in targets:
            next_values = attraction[j, valid[j]]
            if next_values.size:
                k = max(1, int(np.ceil(np.sqrt(next_values.size))))
                continuation[i, j] = np.mean(np.sort(next_values)[-k:])

    for j in range(n):
        sources = np.flatnonzero(valid[:, j])
        for i in sources:
            previous_values = attraction[valid[:, i], i]
            if previous_values.size:
                k = max(1, int(np.ceil(np.sqrt(previous_values.size))))
                predecessor[i, j] = np.mean(
                    np.sort(previous_values)[-k:]
                )

    long_term = np.sqrt(
        np.clip(continuation * predecessor, 0.0, 1.0)
    )

    forward_potential = np.sqrt(
        np.clip(
            row_best_attraction[:, None] * col_best_attraction[None, :],
            0.0,
            1.0,
        )
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

    nearest = np.argmin(safe_distances, axis=1)
    mutual_nearest = np.zeros_like(attraction)
    for i in range(n):
        j = int(nearest[i])
        if (
            j != i
            and valid[i, j]
            and valid[j, i]
            and int(nearest[j]) == i
        ):
            mutual_nearest[i, j] = 1.0

    local_distance = distance_score * (
        0.88 + 0.12 * local_score
    )

    scores = (
        0.30 * attraction
        + 0.25 * local_distance
        + 0.13 * rank_signal
        + 0.17 * long_term
        + 0.08 * forward_potential
        + 0.05 * edge_diversity
        + 0.02 * mutual
    )

    scores *= 0.94 + 0.06 * np.sqrt(
        np.clip(edge_diversity, 0.0, 1.0)
    )
    scores *= 1.0 + 0.04 * mutual_nearest

    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    return scores.astype(float, copy=False)
