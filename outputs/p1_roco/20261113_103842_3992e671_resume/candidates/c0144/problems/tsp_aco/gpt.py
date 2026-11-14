import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distance_matrix)
    distances = np.where(
        finite,
        np.maximum(distance_matrix, 0.0),
        np.inf,
    ).copy()
    np.fill_diagonal(distances, np.inf)

    valid = np.isfinite(distances)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    row_nearest = np.min(distances, axis=1)
    col_nearest = np.min(distances, axis=0)

    finite_nearest = np.concatenate(
        [
            row_nearest[np.isfinite(row_nearest)],
            col_nearest[np.isfinite(col_nearest)],
        ]
    )
    if finite_nearest.size == 0:
        return np.zeros((n, n), dtype=float)

    scale = float(np.median(finite_nearest))
    if not np.isfinite(scale) or scale <= 0.0:
        finite_values = distances[np.isfinite(distances)]
        scale = float(np.median(finite_values)) if finite_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    # Multi-scale global desirability. Each term is monotone in the raw distance.
    global_score = np.zeros((n, n), dtype=float)
    for factor in (0.5, 1.0, 2.0):
        global_score += np.exp(-distances / (factor * scale))
    global_score /= 3.0

    # Outgoing and incoming local rank signals.
    outgoing_rank = np.zeros((n, n), dtype=float)
    incoming_rank = np.zeros((n, n), dtype=float)

    for i in range(n):
        indices = np.flatnonzero(np.isfinite(distances[i]))
        if indices.size == 0:
            continue
        ordered = indices[np.argsort(distances[i, indices], kind="stable")]
        if ordered.size == 1:
            outgoing_rank[i, ordered[0]] = 1.0
        else:
            outgoing_rank[i, ordered] = 1.0 - (
                np.arange(ordered.size, dtype=float) / (ordered.size - 1)
            )

    for j in range(n):
        indices = np.flatnonzero(np.isfinite(distances[:, j]))
        if indices.size == 0:
            continue
        ordered = indices[np.argsort(distances[ordered if False else indices, j], kind="stable")]
        if ordered.size == 1:
            incoming_rank[ordered[0], j] = 1.0
        else:
            incoming_rank[ordered, j] = 1.0 - (
                np.arange(ordered.size, dtype=float) / (ordered.size - 1)
            )

    local_rank = 0.5 * (outgoing_rank + incoming_rank)

    # Mutual nearest-neighbor support.
    nearest_out = np.argmin(distances, axis=1)
    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_out[i]
        if np.isfinite(distances[i, j]) and nearest_out[j] == i:
            mutual[i, j] = 1.0

    # Multi-scale endpoint reachability from outgoing and incoming neighborhoods.
    node_reach = np.zeros(n, dtype=float)
    neighborhood_sizes = sorted(
        set([1, max(1, int(np.sqrt(n - 1))), n - 1])
    )

    for node in range(n):
        outgoing = np.sort(distances[node, np.isfinite(distances[node])])
        incoming = np.sort(distances[np.isfinite(distances[:, node]), node])

        values = []
        for k in neighborhood_sizes:
            parts = []
            if outgoing.size:
                parts.append(outgoing[: min(k, outgoing.size)].mean())
            if incoming.size:
                parts.append(incoming[: min(k, incoming.size)].mean())
            if parts:
                values.append(1.0 / (1.0 + float(np.mean(parts)) / scale))

        if values:
            node_reach[node] = float(np.mean(values))

    reachability = 0.5 * (
        node_reach[:, None] + node_reach[None, :]
    )

    # Best two-step completion through an intermediate node.
    two_step = np.zeros((n, n), dtype=float)
    if n > 2:
        for i in range(n):
            via = distances[i, :, None] + distances[None, :, :]
            via[:, i] = np.inf
            via[i, :] = np.inf
            completion = np.min(via, axis=0)
            completion[i] = np.inf
            finite_completion = np.isfinite(completion)
            two_step[i, finite_completion] = np.exp(
                -completion[finite_completion] / (2.0 * scale)
            )

    # Mild long-term estimate based on the edge and endpoint access costs.
    endpoint_cost = row_nearest[:, None] + col_nearest[None, :]
    long_term = np.zeros((n, n), dtype=float)
    long_term[valid] = np.exp(
        -0.25
        * (distances[valid] + endpoint_cost[valid])
        / scale
    )

    # Additive blending limits interaction distortion and preserves exploration.
    scores = (
        0.46 * global_score
        + 0.22 * local_rank
        + 0.10 * mutual
        + 0.08 * reachability
        + 0.08 * two_step
        + 0.06 * long_term
    )

    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0

    maximum = float(scores.max(initial=0.0))
    if maximum > 0.0:
        scores /= maximum

    return scores
