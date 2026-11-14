import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, diverse, long-horizon priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    values = distances[valid]
    median = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr = max(float(q75 - q25), np.finfo(float).eps)

    scale = max(median, iqr, np.finfo(float).eps)
    cap = max(float(q75 + 4.0 * scale), float(np.max(values)))
    clipped = np.minimum(distances, cap)

    normalized = clipped / (scale + np.finfo(float).eps)

    attraction = 1.0 / (1.0 + normalized)
    attraction[~valid] = 0.0

    global_scale = max(float(np.percentile(values, 75.0)), scale)
    potential = np.exp(-clipped / (global_scale + np.finfo(float).eps))
    potential[~valid] = 0.0

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        order = indices[np.argsort(clipped[i, indices], kind="mergesort")]
        if order.size == 1:
            row_rank[i, order[0]] = 1.0
        else:
            row_rank[i, order] = 1.0 - (
                np.arange(order.size, dtype=float) / (order.size - 1)
            )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue

        order = indices[np.argsort(clipped[indices, j], kind="mergesort")]
        if order.size == 1:
            col_rank[order[0], j] = 1.0
        else:
            col_rank[order, j] = 1.0 - (
                np.arange(order.size, dtype=float) / (order.size - 1)
            )

    mutual_rank = np.sqrt(row_rank * col_rank)

    outgoing_reserve = np.zeros(n, dtype=float)
    incoming_reserve = np.zeros(n, dtype=float)

    for i in range(n):
        outgoing = attraction[i, valid[i]]
        incoming = attraction[valid[:, i], i]

        if outgoing.size:
            take = min(5, outgoing.size)
            outgoing_reserve[i] = float(np.mean(np.partition(outgoing, -take)[-take:]))

        if incoming.size:
            take = min(5, incoming.size)
            incoming_reserve[i] = float(np.mean(np.partition(incoming, -take)[-take:]))

    reserve_scale = max(
        float(np.max(outgoing_reserve)),
        float(np.max(incoming_reserve)),
        np.finfo(float).eps,
    )

    future_bridge = np.sqrt(
        incoming_reserve[:, None] * outgoing_reserve[None, :]
    ) / reserve_scale

    future_bridge[~valid] = 0.0

    base = (
        0.30 * attraction
        + 0.16 * potential
        + 0.16 * row_rank
        + 0.14 * col_rank
        + 0.12 * mutual_rank
        + 0.12 * future_bridge
    )
    base[~valid] = 0.0

    row_mass = np.sum(base, axis=1, keepdims=True)
    col_mass = np.sum(base, axis=0, keepdims=True)

    balanced = base / np.sqrt(
        (row_mass + np.finfo(float).eps)
        * (col_mass + np.finfo(float).eps)
    )
    balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if balanced_max > 0.0:
        balanced /= balanced_max

    endpoint_opportunity = 1.0 / np.sqrt(
        (row_mass + np.finfo(float).eps)
        * (col_mass + np.finfo(float).eps)
    )
    opportunity_max = float(np.max(endpoint_opportunity))
    if opportunity_max > 0.0:
        endpoint_opportunity /= opportunity_max

    score = (
        0.68 * base
        + 0.20 * balanced
        + 0.12 * endpoint_opportunity
    )
    score[~valid] = 0.0

    candidate_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(max(1, n - 1)))))
    )

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size <= candidate_count:
            continue

        order = candidates[
            np.argsort(-score[i, candidates], kind="mergesort")
        ]

        gate = np.full(n, 0.18, dtype=float)
        gate[order[:candidate_count]] = 1.0

        rare_edges = endpoint_opportunity[i] > 0.5
        gate[rare_edges & valid[i]] = np.maximum(
            gate[rare_edges & valid[i]],
            0.28,
        )

        score[i] *= gate

    score[~valid] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        result = score / maximum

    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return result
