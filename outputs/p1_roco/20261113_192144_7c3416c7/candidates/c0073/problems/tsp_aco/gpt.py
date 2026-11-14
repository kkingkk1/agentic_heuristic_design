import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, globally diverse, long-horizon TSP edge priors."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    finite_values = distances[valid]
    offset = float(np.min(finite_values))
    shifted = np.maximum(finite_values - offset, 0.0)

    q25, q50, q75 = np.percentile(shifted, [25.0, 50.0, 75.0])
    iqr = max(float(q75 - q25), np.finfo(float).eps)
    robust_scale = max(float(q50), iqr, np.finfo(float).eps)

    # Apply actual Winsor-style truncation instead of merely computing a bound.
    cap = max(float(q75 + 4.0 * robust_scale), float(q75))
    clipped_values = np.minimum(shifted, cap)

    costs = np.full((n, n), cap, dtype=float)
    costs[valid] = clipped_values

    eps = np.finfo(float).eps * max(1.0, robust_scale)
    normalized_cost = costs / (robust_scale + eps)

    attraction = 1.0 / (1.0 + normalized_cost)
    attraction[~valid] = 0.0

    global_scale = max(float(np.percentile(clipped_values, 75.0)), robust_scale)
    global_potential = np.exp(-costs / (global_scale + eps))
    global_potential[~valid] = 0.0

    row_rank = np.zeros((n, n), dtype=float)
    col_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(valid[i])
        if indices.size == 0:
            continue

        order = indices[
            np.argsort(costs[i, indices], kind="mergesort")
        ]
        if order.size == 1:
            row_rank[i, order[0]] = 1.0
        else:
            row_rank[i, order] = 1.0 - (
                np.arange(order.size, dtype=float) / float(order.size - 1)
            )

    for j in range(n):
        indices = np.flatnonzero(valid[:, j])
        if indices.size == 0:
            continue

        order = indices[
            np.argsort(costs[indices, j], kind="mergesort")
        ]
        if order.size == 1:
            col_rank[order[0], j] = 1.0
        else:
            col_rank[order, j] = 1.0 - (
                np.arange(order.size, dtype=float) / float(order.size - 1)
            )

    mutual_quality = np.sqrt(row_rank * col_rank)
    mutual_quality[~valid] = 0.0

    # Reserve quality estimates the availability of good continuation edges.
    outgoing_reserve = np.zeros(n, dtype=float)
    incoming_reserve = np.zeros(n, dtype=float)

    for i in range(n):
        outgoing = attraction[i, valid[i]]
        incoming = attraction[valid[:, i], i]

        if outgoing.size:
            take = min(4, outgoing.size)
            outgoing_reserve[i] = float(
                np.mean(np.sort(outgoing)[-take:])
            )

        if incoming.size:
            take = min(4, incoming.size)
            incoming_reserve[i] = float(
                np.mean(np.sort(incoming)[-take:])
            )

    reserve_scale = max(
        float(np.max(outgoing_reserve)),
        float(np.max(incoming_reserve)),
        eps,
    )

    # For edge i -> j, preserve both the future after j and the predecessor
    # availability before i.
    long_horizon = np.sqrt(
        np.outer(incoming_reserve, outgoing_reserve)
    ) / reserve_scale
    long_horizon[~valid] = 0.0

    base = (
        0.30 * attraction
        + 0.18 * row_rank
        + 0.14 * col_rank
        + 0.14 * global_potential
        + 0.14 * mutual_quality
        + 0.10 * long_horizon
    )
    base[~valid] = 0.0

    # Normalize row and column mass to discourage a few destination hubs
    # or source nodes from monopolizing the prior.
    row_mass = np.sum(base, axis=1, keepdims=True)
    col_mass = np.sum(base, axis=0, keepdims=True)
    balanced = base / np.sqrt(
        (row_mass + eps) * (col_mass + eps)
    )
    balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if balanced_max > 0.0:
        balanced = balanced / balanced_max

    score = 0.72 * base + 0.28 * balanced
    score[~valid] = 0.0

    # Soft candidate gating: retain all legal edges with an exploration floor.
    candidate_count = min(
        n - 1,
        max(3, int(np.ceil(np.sqrt(max(1, n - 1)))))
    )

    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        if candidates.size > candidate_count:
            order = candidates[
                np.argsort(-score[i, candidates], kind="mergesort")
            ]
            gate = np.full(n, 0.18, dtype=float)
            gate[order[:candidate_count]] = 1.0
            score[i] *= gate

    score[~valid] = 0.0
    maximum = float(np.max(score))

    if maximum > 0.0:
        result = score / maximum

    np.fill_diagonal(result, 0.0)
    result[~valid] = 0.0
    return result
