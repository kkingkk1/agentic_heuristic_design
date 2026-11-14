import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")

    n = distance_matrix.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps

    d = np.maximum(distance_matrix, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)

    edge_values = d[valid]
    if edge_values.size == 0:
        return np.zeros((n, n), dtype=float)

    low = float(np.percentile(edge_values, 10.0))
    high = float(np.percentile(edge_values, 90.0))

    raw_min = float(np.min(edge_values))
    raw_max = float(np.max(edge_values))

    if high - low > eps:
        clipped = np.clip(d, low, high)
        distance_quality = (high - clipped) / (high - low)
    elif raw_max - raw_min > eps:
        distance_quality = (raw_max - d) / (raw_max - raw_min)
    else:
        distance_quality = np.ones((n, n), dtype=float)

    distance_quality = np.where(valid, distance_quality, 0.0)
    np.fill_diagonal(distance_quality, 0.0)

    def normalize_valid(matrix: np.ndarray) -> np.ndarray:
        result = np.zeros_like(matrix, dtype=float)
        values = matrix[valid]

        if values.size == 0:
            return result

        minimum = float(np.min(values))
        maximum = float(np.max(values))

        if maximum - minimum <= eps:
            result[valid] = 1.0
        else:
            result[valid] = (
                matrix[valid] - minimum
            ) / (maximum - minimum)

        return result

    # Preserve a global ordering signal without repeatedly rescaling every component.
    flat_quality = distance_quality[valid]
    order = np.argsort(-flat_quality, kind="mergesort")
    global_rank_values = np.zeros_like(flat_quality, dtype=float)

    if flat_quality.size == 1:
        global_rank_values[:] = 1.0
    else:
        global_rank_values[order] = (
            1.0 - np.arange(flat_quality.size, dtype=float)
            / float(flat_quality.size - 1)
        )

    global_rank = np.zeros((n, n), dtype=float)
    global_rank[valid] = global_rank_values

    # Local quality mass and entropy estimate how many good alternatives
    # each endpoint has. High entropy means less dependence on one edge.
    row_mass = np.sum(distance_quality, axis=1)
    row_mass = np.maximum(row_mass, eps)

    probabilities = distance_quality / row_mass[:, None]
    node_diversity = np.zeros(n, dtype=float)

    for i in range(n):
        row = probabilities[i, valid[i]]
        degree = row.size

        if degree <= 1:
            node_diversity[i] = 1.0
            continue

        positive = row[row > eps]
        if positive.size == 0:
            node_diversity[i] = 1.0
        else:
            entropy = -float(np.sum(positive * np.log(positive)))
            node_diversity[i] = entropy / np.log(float(degree))

    endpoint_diversity = np.sqrt(
        node_diversity[:, None] * node_diversity[None, :]
    )
    endpoint_diversity = np.where(valid, endpoint_diversity, 0.0)

    # Residual alternative quality after removing the current edge.
    # This directly measures whether the edge leaves both endpoints extensible.
    residual_support = np.zeros((n, n), dtype=float)

    for i in range(n):
        row = distance_quality[i]
        total = float(np.sum(row))

        if total <= eps:
            continue

        for j in np.flatnonzero(valid[i]):
            residual_support[i, j] = max(
                total - float(distance_quality[i, j]),
                0.0,
            ) / total

    residual_support = 0.5 * (
        residual_support + residual_support.T
    )

    # Long-term potential: average of the strongest alternatives at each
    # endpoint, excluding the candidate edge itself.
    continuation = np.zeros((n, n), dtype=float)
    width = min(3, max(n - 2, 0))

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        values = distance_quality[i, candidates]

        if candidates.size == 0:
            continue

        sorted_values = np.sort(values)[::-1]

        for j in candidates:
            alternatives = values[candidates != j]

            if alternatives.size == 0:
                continuation[i, j] = 0.0
            else:
                k = min(width, alternatives.size)
                if k <= 0:
                    continuation[i, j] = float(np.mean(alternatives))
                else:
                    best = np.sort(alternatives)[-k:]
                    continuation[i, j] = float(np.mean(best))

    continuation = 0.5 * (
        continuation + continuation.T
    )

    # Geometric aggregation requires both endpoints to have future options.
    continuation_left = np.maximum(continuation, 0.0)
    long_term = np.sqrt(
        continuation_left * continuation_left.T
    )

    # Weak two-hop support. It rewards globally connected edges without
    # allowing indirect structure to override direct distance quality.
    transition = distance_quality / row_mass[:, None]
    two_hop = transition @ transition.T
    two_hop = np.maximum(two_hop, 0.0)

    # Mild penalty for edges that monopolize either endpoint.
    edge_share = distance_quality / row_mass[:, None]
    anti_monopoly = 1.0 - np.sqrt(
        np.clip(edge_share * edge_share.T, 0.0, 1.0)
    )
    anti_monopoly = 0.5 + 0.5 * anti_monopoly

    score = (
        0.50 * distance_quality
        + 0.16 * global_rank
        + 0.12 * endpoint_diversity
        + 0.10 * residual_support
        + 0.09 * long_term
        + 0.03 * anti_monopoly
        + 0.02 * normalize_valid(two_hop)
    )

    score = 0.5 * (score + score.T)
    score = np.where(valid, score, 0.0)
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    values = score[valid]

    if values.size == 0:
        return np.zeros((n, n), dtype=float)

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if maximum - minimum <= eps:
        score[valid] = 1.0
    else:
        score[valid] = (
            score[valid] - minimum
        ) / (maximum - minimum)

    maximum = float(np.max(score))
    if maximum > eps:
        score /= maximum

    score = np.where(valid, score, 0.0)
    np.fill_diagonal(score, 0.0)

    return np.asarray(score, dtype=float)
