import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term edge-priority indicators for TSP."""

    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(d)
    valid = finite & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    reciprocal = valid & valid.T
    symmetric = np.where(
        reciprocal,
        0.5 * (d + d.T),
        np.where(valid, d, np.where(valid.T, d.T, np.inf)),
    )
    symmetric[~(valid | valid.T)] = np.inf
    np.fill_diagonal(symmetric, np.inf)

    finite_sym = symmetric[np.isfinite(symmetric)]
    if finite_sym.size == 0:
        return result

    positive = finite_sym[finite_sym > eps]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        global_scale = max(float(np.median(finite_sym)), 1.0)
    global_scale = max(global_scale, eps)

    degree = max(1, n - 1)
    ordered = np.sort(symmetric, axis=1)

    local_k = max(1, min(degree, int(np.ceil(np.sqrt(float(n))))))
    local_neighbors = ordered[:, :local_k]
    local_scale = np.nanmedian(
        np.where(np.isfinite(local_neighbors), local_neighbors, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = (
        0.64 * np.sqrt(local_scale[:, None] * local_scale[None, :])
        + 0.36 * global_scale
    )
    pair_scale = np.maximum(pair_scale, eps)

    relative = np.divide(
        symmetric,
        pair_scale,
        out=np.full_like(symmetric, np.inf),
        where=np.isfinite(symmetric),
    )

    distance_quality = np.exp(-np.clip(relative, 0.0, 40.0))
    distance_quality[~valid] = 0.0

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.full((n, n), np.inf, dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_temperature = max(1.0, 0.9 * np.sqrt(float(n)))
    outgoing_rank = np.exp(-row_rank / rank_temperature)
    incoming_rank = np.exp(-col_rank / rank_temperature)
    outgoing_rank[~valid] = 0.0
    incoming_rank[~valid] = 0.0

    mutual_rank = np.sqrt(outgoing_rank * incoming_rank)

    reciprocal_k = max(
        1,
        min(degree, int(np.ceil(np.log2(float(n) + 1.0)))),
    )
    mutual_edge = (
        valid
        & (row_rank < reciprocal_k)
        & (col_rank < reciprocal_k)
    )

    local_quality = distance_quality * (
        0.60 + 0.40 * mutual_rank
    )
    local_quality *= 1.0 + 0.22 * mutual_edge.astype(float)
    local_quality[~valid] = 0.0
    local_quality = np.clip(local_quality, 0.0, 1.0)

    # Candidate-set statistics. Multiple good alternatives increase
    # long-term potential, while candidate entropy prevents brittle
    # dependence on a single nearest neighbor.
    candidate_k = max(
        1,
        min(degree, int(np.ceil(1.5 * np.sqrt(float(n))))),
    )
    candidate_temperature = max(
        0.035,
        0.16 / np.sqrt(float(n)),
    )

    def candidate_statistics(values: np.ndarray, axis: int):
        if axis == 1:
            ordered_values = np.sort(values, axis=1)[:, ::-1]
        else:
            ordered_values = np.sort(values, axis=0)[::-1, :]

        top = ordered_values[:candidate_k] if axis == 0 else ordered_values[:, :candidate_k]
        top = np.where(np.isfinite(top), top, 0.0)

        maximum = np.max(top, axis=axis, keepdims=True)
        weights = np.exp(
            np.clip((top - maximum) / candidate_temperature, -50.0, 0.0)
        )
        weights *= top > 0.0

        denominator = np.sum(weights, axis=axis, keepdims=True)
        normalized = weights / np.maximum(denominator, eps)

        weighted_quality = np.sum(
            normalized * top,
            axis=axis,
        )
        support_count = np.sum(top > 0.0, axis=axis)
        weighted_quality *= np.minimum(
            support_count / float(candidate_k),
            1.0,
        )

        entropy = -np.sum(
            normalized * np.log(np.maximum(normalized, eps)),
            axis=axis,
        )
        entropy /= max(np.log(float(candidate_k) + 1.0), 1.0)

        potential = weighted_quality * (0.72 + 0.28 * entropy)
        return np.clip(potential, 0.0, 1.0)

    outgoing_core = candidate_statistics(local_quality, axis=1)
    incoming_core = candidate_statistics(local_quality.T, axis=1)

    # Two-step continuation: an edge is valuable when its destination
    # retains good outgoing extensions and its source remains reconnectable.
    outgoing_projection = np.zeros(n, dtype=float)
    incoming_projection = np.zeros(n, dtype=float)

    for node in range(n):
        outgoing_edges = local_quality[node]
        outgoing_projection[node] = np.sum(
            outgoing_edges * outgoing_core
        ) / max(float(np.count_nonzero(outgoing_edges)), 1.0)

        incoming_edges = local_quality[:, node]
        incoming_projection[node] = np.sum(
            incoming_edges * incoming_core
        ) / max(float(np.count_nonzero(incoming_edges)), 1.0)

    outgoing_projection = np.clip(
        0.55 * outgoing_core + 0.45 * outgoing_projection,
        0.0,
        1.0,
    )
    incoming_projection = np.clip(
        0.55 * incoming_core + 0.45 * incoming_projection,
        0.0,
        1.0,
    )

    continuation = np.sqrt(
        outgoing_projection[None, :] * incoming_projection[:, None]
    )
    continuation = np.clip(continuation, 0.0, 1.0)

    # Protect nodes whose neighborhood has a pronounced distance gap.
    second = ordered[:, min(1, degree - 1)]
    kth = ordered[:, local_k - 1]
    second = np.where(np.isfinite(second), second, global_scale)
    kth = np.where(np.isfinite(kth), kth, global_scale)

    scarcity = np.clip(
        (kth - local_scale) / np.maximum(kth, eps),
        0.0,
        1.0,
    )
    regret = np.clip(
        (second - local_scale) / np.maximum(second, eps),
        0.0,
        1.0,
    )

    edge_score = local_quality
    edge_score *= 0.84 + 0.30 * continuation
    edge_score *= 1.0 + 0.16 * (
        scarcity[:, None] + scarcity[None, :]
    )
    edge_score *= 1.0 + 0.10 * (
        regret[:, None] + regret[None, :]
    )

    # Suppress extreme long edges without deleting them entirely.
    edge_score *= np.exp(
        -0.30 * np.maximum(relative - 1.0, 0.0)
    )
    edge_score[~valid] = 0.0
    edge_score = np.nan_to_num(
        edge_score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if not np.any(edge_score > 0.0):
        count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / max(count, 1.0)
        return result

    # Damped capacity correction. It raises underrepresented rows and
    # columns, but its bounded exponent avoids flattening quality gaps.
    row_mass = edge_score.sum(axis=1)
    col_mass = edge_score.sum(axis=0)

    positive_row = row_mass[row_mass > 0.0]
    positive_col = col_mass[col_mass > 0.0]
    reference_row = (
        float(np.median(positive_row))
        if positive_row.size
        else 1.0
    )
    reference_col = (
        float(np.median(positive_col))
        if positive_col.size
        else 1.0
    )

    row_need = np.sqrt(
        reference_row / np.maximum(row_mass, eps)
    )
    col_need = np.sqrt(
        reference_col / np.maximum(col_mass, eps)
    )

    row_need = np.clip(row_need, 0.70, 1.80)
    col_need = np.clip(col_need, 0.70, 1.80)

    capacity_factor = np.sqrt(
        row_need[:, None] * col_need[None, :]
    )
    capacity_factor /= max(
        float(np.mean(capacity_factor[valid])),
        eps,
    )

    calibrated = edge_score * (
        0.90 + 0.10 * np.clip(capacity_factor, 0.55, 1.75)
    )
    calibrated[~valid] = 0.0

    # Soft competition penalty prevents a few dominant edges from
    # absorbing all probability mass while preserving ranking gaps.
    row_share = calibrated / np.maximum(
        calibrated.sum(axis=1, keepdims=True),
        eps,
    )
    col_share = calibrated / np.maximum(
        calibrated.sum(axis=0, keepdims=True),
        eps,
    )

    dominance = np.sqrt(
        np.maximum(row_share, 0.0)
        * np.maximum(col_share, 0.0)
    )
    dominance = np.clip(dominance, 0.0, 1.0)

    calibrated *= 1.0 - 0.08 * dominance
    calibrated[~valid] = 0.0
    np.fill_diagonal(calibrated, 0.0)

    total = float(calibrated.sum())
    if not np.isfinite(total) or total <= 0.0:
        count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / max(count, 1.0)
        return result

    return calibrated / total
