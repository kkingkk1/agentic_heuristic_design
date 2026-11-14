import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n <= 1:
        return score

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & offdiag
    valid = finite | finite.T

    if not np.any(valid):
        return score

    clipped = np.maximum(raw, 0.0)

    distance = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    distance[both] = 0.5 * (
        clipped[both] + clipped.T[both]
    )
    distance[forward_only] = clipped[forward_only]
    distance[backward_only] = clipped.T[backward_only]
    distance[~valid] = 0.0
    np.fill_diagonal(distance, 0.0)

    observed = distance[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps

    # Main distance affinity.
    temperature = max(0.45 * scale, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -distance[valid] / temperature
    )

    affinity_max = float(np.max(affinity))
    base = affinity / max(affinity_max, eps)

    # Local endpoint scarcity and nearest-neighbour information.
    nearest = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)
    nearest_index = np.full(n, -1, dtype=int)

    for i in range(n):
        row = distance[i].copy()
        row[~valid[i]] = np.inf
        row[i] = np.inf

        order = np.argsort(row, kind="stable")
        finite_order = order[np.isfinite(row[order])]

        if finite_order.size:
            nearest_index[i] = int(finite_order[0])
            nearest[i] = row[finite_order[0]]

        if finite_order.size > 1:
            second[i] = row[finite_order[1]]

    local_gap = second - nearest
    local_gap[~np.isfinite(local_gap)] = 0.20 * scale
    local_gap = np.maximum(local_gap, 0.20 * scale)
    local_gap = np.maximum(local_gap, eps)

    endpoint = np.zeros((n, n), dtype=float)

    for i in range(n):
        valid_row = valid[i]
        if not np.any(valid_row):
            continue

        values = np.exp(
            np.clip(
                -(distance[i, valid_row] - nearest[i]) / local_gap[i],
                -12.0,
                12.0,
            )
        )
        endpoint[i, valid_row] = values

    endpoint = np.sqrt(endpoint * endpoint.T)
    endpoint[~valid] = 0.0

    # Mutual nearest-neighbour protection.
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest_index[i]
        if j >= 0 and nearest_index[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Small local-support bonus based only on restricted nearest-neighbour sets.
    k = min(8, n - 1)
    candidates = np.full((n, k), -1, dtype=int)

    for i in range(n):
        row = distance[i].copy()
        row[~valid[i]] = np.inf
        row[i] = np.inf

        order = np.argsort(row, kind="stable")
        finite_order = order[np.isfinite(row[order])]
        take = min(k, finite_order.size)

        if take:
            candidates[i, :take] = finite_order[:take]

    local_support = np.zeros((n, n), dtype=float)

    for i in range(n):
        neighbors = candidates[i]
        neighbors = neighbors[neighbors >= 0]

        if neighbors.size == 0:
            continue

        threshold = distance[i, neighbors[-1]]
        local_edges = valid[i] & (distance[i] <= threshold)
        local_count = np.count_nonzero(local_edges)

        if local_count > 0:
            local_support[i, local_edges] = 1.0 / local_count

    local_support = local_support + local_support.T
    support_max = float(np.max(local_support))

    if support_max > 0.0 and np.isfinite(support_max):
        local_support = local_support / support_max

    # Restricted two-hop replacement quality.
    best_bypass = np.full((n, n), np.inf, dtype=float)

    for column in range(k):
        middles = candidates[:, column]
        middles = middles[middles >= 0]

        for middle in middles:
            left = valid[:, middle]
            right = valid[middle, :]

            feasible = left[:, None] & right[None, :]
            route = (
                distance[:, middle, None]
                + distance[None, middle, :]
            )
            candidate = np.where(feasible, route, np.inf)
            best_bypass = np.minimum(best_bypass, candidate)

    measurable = valid & np.isfinite(best_bypass)
    bypass_gain = np.zeros((n, n), dtype=float)
    bypass_gain[measurable] = np.maximum(
        best_bypass[measurable] - distance[measurable],
        0.0,
    )

    irreplaceability = np.zeros((n, n), dtype=float)
    irreplaceability[measurable] = 1.0 - np.exp(
        -bypass_gain[measurable] / max(0.85 * scale, eps)
    )

    # Conservative adaptive fusion:
    # strong local edges receive a small support correction, while the
    # distance affinity remains the dominant signal.
    local_gate = np.clip(
        0.5 * endpoint + 0.5 * local_support,
        0.0,
        1.0,
    )

    score = (
        0.62 * base
        + 0.21 * endpoint
        + 0.10 * irreplaceability
        + 0.05 * mutual
        + 0.02 * local_gate
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))

    if np.isfinite(maximum) and maximum > 0.0:
        score = score / maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
