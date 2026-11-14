import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally normalized symmetric edge-priority indicators for TSP."""

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

    symmetric = np.full((n, n), np.inf, dtype=float)

    both = directed_valid & directed_valid.T
    forward_only = directed_valid & ~directed_valid.T
    backward_only = ~directed_valid & directed_valid.T

    symmetric[both] = 0.5 * (d[both] + d.T[both])
    symmetric[forward_only] = d[forward_only]
    symmetric[backward_only] = d.T[backward_only]

    np.fill_diagonal(symmetric, np.inf)

    finite_values = symmetric[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = 1.0
    else:
        return result

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    local_scale = np.full(n, global_scale, dtype=float)
    best = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)
    ranks = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(np.isfinite(symmetric[i]))
        if neighbors.size == 0:
            continue

        order = neighbors[
            np.argsort(symmetric[i, neighbors], kind="stable")
        ]
        values = symmetric[i, order]

        ranks[i, order] = np.arange(order.size, dtype=float)
        best[i] = values[0]
        second[i] = values[1] if values.size > 1 else values[0]

        k = min(3, values.size)
        local_values = values[:k]
        positive_local = local_values[local_values > 0.0]

        if positive_local.size:
            local_scale[i] = max(float(np.median(positive_local)), eps)
        else:
            local_scale[i] = global_scale

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    regret = np.divide(
        second - best,
        np.maximum(second, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second),
    )
    regret = np.clip(regret, 0.0, 1.0)

    endpoint_urgency = np.sqrt(
        (1.0 + 0.35 * regret)[:, None]
        * (1.0 + 0.35 * regret)[None, :]
    )

    distance_quality = np.exp(
        -np.minimum(symmetric / global_scale, 60.0)
    )

    local_quality = np.exp(
        -np.minimum(
            symmetric
            / np.sqrt(local_scale[:, None] * local_scale[None, :]),
            60.0,
        )
    )

    rank_quality = 1.0 / (1.0 + ranks + ranks.T)

    nearest_count = min(3, n - 1)
    mutual = (
        (ranks < nearest_count)
        & (ranks.T < nearest_count)
    ).astype(float)
    mutual_bonus = 1.0 + 0.20 * mutual

    base_score = (
        distance_quality ** 0.60
        * local_quality ** 0.25
        * rank_quality ** 0.15
        * mutual_bonus
        * endpoint_urgency
    )

    finite = np.isfinite(symmetric) & valid
    score = np.zeros((n, n), dtype=float)
    score[finite] = base_score[finite]

    # 保留少量全局保底，避免跨簇桥接边因局部尺度被完全压制。
    bridge_floor = 0.035 * distance_quality
    score[finite] = np.maximum(score[finite], bridge_floor[finite])

    # 对每个端点的前两条边增加轻微候选保留，防止稀疏节点失去可行连接。
    reserve_count = min(2, n - 1)
    for i in range(n):
        candidates = np.flatnonzero(np.isfinite(symmetric[i]))
        if candidates.size == 0:
            continue

        ordered = candidates[
            np.argsort(symmetric[i, candidates], kind="stable")
        ]
        reserved = ordered[:reserve_count]

        for j in reserved:
            if finite[i, j]:
                reserve = 0.08 * distance_quality[i, j]
                score[i, j] = max(score[i, j], reserve)

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    score = 0.5 * (score + score.T)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        result = score / maximum

    return result
