import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & offdiag
    valid = finite | finite.T

    if not np.any(valid):
        return result

    nonnegative = np.maximum(raw, 0.0)
    distances = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    only_forward = finite & ~finite.T
    only_backward = ~finite & finite.T

    distances[both] = 0.5 * (
        nonnegative[both] + nonnegative.T[both]
    )

    # Copy one-sided finite edges directly to both orientations.
    distances[only_forward] = nonnegative[only_forward]
    distances[only_backward] = nonnegative.T[only_backward]

    distances[~valid] = 0.0
    np.fill_diagonal(distances, 0.0)

    observed = distances[valid]
    positive = observed[observed > 0.0]

    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    # Distance affinity.
    temperature = max(0.45 * scale, np.finfo(float).eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-distances[valid] / temperature)

    # Nearest and second-nearest distances for endpoint scarcity.
    nearest_distance = np.full(n, np.inf, dtype=float)
    second_distance = np.full(n, np.inf, dtype=float)
    nearest_index = np.full(n, -1, dtype=int)

    for i in range(n):
        row = distances[i].copy()
        row[~valid[i]] = np.inf
        row[i] = np.inf

        order = np.argsort(row, kind="stable")
        finite_order = order[np.isfinite(row[order])]

        if finite_order.size:
            nearest_index[i] = int(finite_order[0])
            nearest_distance[i] = row[finite_order[0]]

        if finite_order.size > 1:
            second_distance[i] = row[finite_order[1]]

    local_scale = second_distance - nearest_distance
    local_scale[~np.isfinite(local_scale)] = 0.20 * scale
    local_scale = np.maximum(local_scale, 0.20 * scale)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    endpoint = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        values = np.exp(
            np.clip(
                (second_distance[i] - distances[i, row_valid])
                / local_scale[i],
                -12.0,
                12.0,
            )
        )

        # If no second neighbour exists, avoid treating every edge as
        # maximally exclusive solely because of an infinite sentinel.
        if not np.isfinite(second_distance[i]):
            values = np.exp(
                np.clip(
                    (nearest_distance[i] - distances[i, row_valid])
                    / local_scale[i],
                    -12.0,
                    12.0,
                )
            )

        endpoint[i, row_valid] = values

    endpoint = np.sqrt(endpoint * endpoint.T)
    endpoint[~valid] = 0.0

    # Mutual-nearest-neighbour protection.
    mutual = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest_index[i]
        if j >= 0 and nearest_index[j] == i:
            mutual[i, j] = 1.0
            mutual[j, i] = 1.0

    # Limited nearest-neighbour sets for efficient two-hop replacement checks.
    k = min(8, n - 1)
    candidate_indices = np.full((n, k), -1, dtype=int)

    for i in range(n):
        row = distances[i].copy()
        row[~valid[i]] = np.inf
        row[i] = np.inf

        order = np.argsort(row, kind="stable")
        finite_order = order[np.isfinite(row[order])]
        take = min(k, finite_order.size)

        if take:
            candidate_indices[i, :take] = finite_order[:take]

    best_bypass = np.full((n, n), np.inf, dtype=float)

    for column in range(k):
        middles = candidate_indices[:, column]

        for middle in middles[middles >= 0]:
            left_valid = valid[:, middle]
            right_valid = valid[middle, :]

            route = distances[:, middle, None] + distances[None, middle, :]
            feasible = left_valid[:, None] & right_valid[None, :]

            candidate = np.where(feasible, route, np.inf)
            best_bypass = np.minimum(best_bypass, candidate)

    # Reward edges whose best restricted two-hop replacement is substantially
    # longer, but do not reward edges without a measured replacement.
    measurable = valid & np.isfinite(best_bypass)
    bypass_gain = np.zeros((n, n), dtype=float)
    bypass_gain[measurable] = np.maximum(
        best_bypass[measurable] - distances[measurable],
        0.0,
    )

    irreplaceability = np.zeros((n, n), dtype=float)
    irreplaceability[measurable] = 1.0 - np.exp(
        -bypass_gain[measurable]
        / max(0.85 * scale, np.finfo(float).eps)
    )

    affinity_max = float(np.max(affinity))
    base = affinity / max(affinity_max, np.finfo(float).eps)

    score = (
        0.62 * base
        + 0.23 * endpoint
        + 0.10 * irreplaceability
        + 0.05 * mutual
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    # The construction is already symmetric; no later averaging is used.
    maximum = float(np.max(score))

    if np.isfinite(maximum) and maximum > 0.0:
        score = score / maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
