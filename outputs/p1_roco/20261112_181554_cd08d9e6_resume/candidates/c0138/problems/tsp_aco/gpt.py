import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""

    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    directed_valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid | directed_valid.T
    if not np.any(valid):
        return result

    distance = np.full((n, n), np.inf, dtype=float)

    both = directed_valid & directed_valid.T
    forward_only = directed_valid & ~directed_valid.T
    backward_only = ~directed_valid & directed_valid.T

    distance[both] = 0.5 * (d[both] + d.T[both])
    distance[forward_only] = d[forward_only]
    distance[backward_only] = d.T[backward_only]
    np.fill_diagonal(distance, np.inf)

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = max(float(np.median(positive_values)), eps)
    elif finite_values.size:
        global_scale = 1.0
    else:
        return result

    ranks = np.full((n, n), np.inf, dtype=float)
    local_scale = np.full(n, global_scale, dtype=float)
    degree = np.zeros(n, dtype=float)
    regret = np.zeros(n, dtype=float)
    quality = np.zeros((n, n), dtype=float)

    neighbor_lists = []

    for i in range(n):
        neighbors = np.flatnonzero(np.isfinite(distance[i]))
        neighbor_lists.append(neighbors)
        degree[i] = float(neighbors.size)

        if neighbors.size == 0:
            continue

        order = neighbors[
            np.argsort(distance[i, neighbors], kind="stable")
        ]
        values = distance[i, order]
        ranks[i, order] = np.arange(order.size, dtype=float)

        best = values[0]
        second = values[1] if values.size > 1 else best
        regret[i] = np.clip(
            (second - best) / max(second, eps),
            0.0,
            1.0,
        )

        positive_local = values[: min(4, values.size)]
        positive_local = positive_local[positive_local > 0.0]
        if positive_local.size:
            local_scale[i] = max(
                float(np.median(positive_local)),
                eps,
            )

    distance_ratio = np.zeros((n, n), dtype=float)
    distance_ratio[valid] = np.minimum(
        distance[valid] / global_scale,
        60.0,
    )

    local_denominator = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    local_ratio = np.zeros((n, n), dtype=float)
    local_ratio[valid] = np.minimum(
        distance[valid] / np.maximum(local_denominator[valid], eps),
        60.0,
    )

    distance_quality = np.zeros((n, n), dtype=float)
    local_quality = np.zeros((n, n), dtype=float)
    distance_quality[valid] = np.exp(-distance_ratio[valid])
    local_quality[valid] = np.exp(-local_ratio[valid])

    # Explicit two-hop support:
    # exclude i and j, and aggregate only paths i -> k -> j.
    two_hop = np.zeros((n, n), dtype=float)
    for i in range(n):
        first_neighbors = neighbor_lists[i]
        if first_neighbors.size == 0:
            continue

        for j in first_neighbors:
            if i == j or not np.isfinite(distance[j, i]):
                continue

            support = 0.0
            for k in first_neighbors:
                if k == i or k == j:
                    continue
                if np.isfinite(distance[k, j]):
                    support += (
                        distance_quality[i, k]
                        * distance_quality[k, j]
                    )

            two_hop[i, j] = support

    access = np.sum(distance_quality, axis=1)
    access_scale = np.maximum(
        np.median(access[access > 0.0])
        if np.any(access > 0.0)
        else 1.0,
        eps,
    )

    # Low access indicates a structurally constrained endpoint.
    access_need = np.zeros(n, dtype=float)
    positive_access = access > 0.0
    access_need[positive_access] = np.clip(
        access_scale / np.maximum(access[positive_access], eps),
        0.5,
        2.5,
    )
    access_need = np.sqrt(access_need)

    max_degree = max(float(n - 1), 1.0)
    sparsity = np.ones(n, dtype=float)
    has_edges = degree > 0.0
    sparsity[has_edges] = np.sqrt(
        np.clip(
            (max_degree + 1.0) / (degree[has_edges] + 1.0),
            1.0,
            max_degree + 1.0,
        )
    )
    sparsity /= max(float(np.max(sparsity)), 1.0)

    # Cap balancing strength on tiny graphs.
    if n <= 4:
        balance_cap = 0.10
    elif n <= 8:
        balance_cap = 0.16
    else:
        balance_cap = 0.24

    endpoint_balance = 1.0 + balance_cap * np.sqrt(
        sparsity[:, None] * sparsity[None, :]
    )

    urgency = np.sqrt(
        (1.0 + 0.24 * regret)[:, None]
        * (1.0 + 0.24 * regret)[None, :]
    )

    rank_quality = 1.0 / (1.0 + ranks + ranks.T)

    nearest_count = min(3, n - 1)
    mutual_nearest = (
        (ranks < nearest_count)
        & (ranks.T < nearest_count)
    ).astype(float)
    mutual_bonus = 1.0 + 0.13 * mutual_nearest

    max_two_hop = float(np.max(two_hop))
    if max_two_hop > 0.0 and np.isfinite(max_two_hop):
        support_quality = np.zeros((n, n), dtype=float)
        support_quality[valid] = np.clip(
            two_hop[valid] / max_two_hop,
            0.0,
            1.0,
        )
    else:
        support_quality = np.zeros((n, n), dtype=float)

    # Low support plus constrained access marks a possible bridge.
    bridge_signal = np.zeros((n, n), dtype=float)
    bridge_signal[valid] = (
        (1.0 - support_quality[valid])
        * np.sqrt(
            access_need[:, None][valid]
            * access_need[None, :][valid]
        )
    )
    bridge_signal = np.clip(bridge_signal, 0.0, 2.0)

    bridge_bonus = np.ones((n, n), dtype=float)
    bridge_bonus[valid] = 1.0 + 0.10 * bridge_signal[valid]

    score = np.zeros((n, n), dtype=float)
    score[valid] = (
        np.power(distance_quality[valid], 0.62)
        * np.power(local_quality[valid], 0.16)
        * np.power(rank_quality[valid], 0.14)
        * (0.92 + 0.08 * support_quality[valid])
        * mutual_bonus[valid]
        * urgency[valid]
        * endpoint_balance[valid]
        * bridge_bonus[valid]
    )

    # Weak adaptive floor: preserve exploration without flattening contrasts.
    endpoint_sparsity = np.sqrt(
        sparsity[:, None] * sparsity[None, :]
    )
    floor_strength = (
        0.012
        + 0.028 * endpoint_sparsity
        + 0.010 * np.maximum(
            sparsity[:, None],
            sparsity[None, :],
        )
    )

    if n <= 4:
        floor_strength *= 0.55

    score[valid] = np.maximum(
        score[valid],
        floor_strength[valid] * distance_quality[valid],
    )

    # Preserve zero-distance edges as the strongest distance-quality edges.
    zero_edges = valid & (distance == 0.0)
    if np.any(zero_edges):
        score[zero_edges] = np.maximum(
            score[zero_edges],
            1.0 + 0.05 * mutual_nearest[zero_edges],
        )

    # Guarantee each endpoint keeps a small number of its best valid links.
    reserve_count = min(2, n - 1)
    for i in range(n):
        neighbors = neighbor_lists[i]
        if neighbors.size == 0:
            continue

        order = neighbors[
            np.argsort(distance[i, neighbors], kind="stable")
        ]

        for j in order[:reserve_count]:
            if not valid[i, j]:
                continue

            reserve = (
                0.035
                * distance_quality[i, j]
                * (0.85 + 0.15 * sparsity[i])
            )
            score[i, j] = max(score[i, j], reserve)

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        result = score / maximum

    return result
