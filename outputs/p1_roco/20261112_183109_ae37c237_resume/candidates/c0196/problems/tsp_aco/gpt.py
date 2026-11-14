import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n < 2:
        return score

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & off_diagonal

    if not np.any(valid):
        return score

    reverse_valid = valid.T
    both_valid = valid & reverse_valid

    pair_distance = np.full((n, n), np.inf, dtype=float)
    pair_distance[both_valid] = (
        d[both_valid] + d.T[both_valid]
    ) * 0.5

    only_forward = valid & ~reverse_valid
    pair_distance[only_forward] = d[only_forward]

    only_reverse = reverse_valid & ~valid
    pair_distance[only_reverse] = d.T[only_reverse]

    np.fill_diagonal(pair_distance, np.inf)

    finite_pair = np.isfinite(pair_distance)
    values = pair_distance[finite_pair]

    positive = values[values > 0.0]
    scale_values = positive if positive.size else values

    if scale_values.size == 0:
        return score

    eps = np.finfo(float).eps
    global_scale = float(np.median(scale_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    k = min(4, max(1, n - 1))
    rank_matrix = np.full((n, n), np.inf, dtype=float)
    local_scale = np.full(n, global_scale, dtype=float)
    local_regret = np.zeros(n, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(np.isfinite(pair_distance[i]))
        if neighbors.size == 0:
            continue

        ordered = neighbors[
            np.argsort(pair_distance[i, neighbors], kind="stable")
        ]
        distances = pair_distance[i, ordered]

        local_scale[i] = float(np.median(distances[:k]))

        if distances.size >= 2:
            best = float(distances[0])
            second = float(distances[1])
            local_regret[i] = np.clip(
                (second - best) / max(second, global_scale, eps),
                0.0,
                1.0,
            )

        rank_matrix[i, ordered] = np.arange(ordered.size, dtype=float)

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(
        0.8 * local_scale + 0.2 * global_scale,
        eps,
    )

    finite_rank = np.isfinite(rank_matrix)
    rank_scale = max(1.0, np.sqrt(float(n)))

    rank_term = np.exp(
        -np.clip(rank_matrix + rank_matrix.T, 0.0, 60.0)
        / rank_scale
    )
    rank_term = np.where(
        finite_rank & finite_rank.T,
        rank_term,
        0.0,
    )
    rank_bonus = 0.45 + 0.55 * rank_term

    mutual = (
        (rank_matrix < k)
        & (rank_matrix.T < k)
    )
    mutual_bonus = 1.0 + 0.52 * mutual.astype(float)

    top_two = (
        (rank_matrix <= 1.0)
        & (rank_matrix.T <= 1.0)
    )
    lookahead_bonus = 1.0 + 0.08 * top_two.astype(float)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        pair_distance,
        pair_scale,
        out=np.full_like(pair_distance, np.inf),
        where=finite_pair,
    )

    global_ratio = np.divide(
        pair_distance,
        global_scale,
        out=np.full_like(pair_distance, np.inf),
        where=finite_pair,
    )

    distance_term = np.exp(
        -np.clip(
            0.72 * relative_distance + 0.28 * global_ratio,
            0.0,
            60.0,
        )
    )

    scarcity_bonus = 1.0 + 0.12 * (
        local_regret[:, None] + local_regret[None, :]
    )

    safe_pair = np.maximum(pair_distance, eps)
    directional_ratio = np.divide(
        d,
        safe_pair,
        out=np.ones_like(d),
        where=valid,
    )
    directional_consistency = np.exp(
        -0.18 * np.clip(directional_ratio - 1.0, 0.0, 20.0)
    )

    core_score = (
        distance_term
        * rank_bonus
        * mutual_bonus
        * lookahead_bonus
        * scarcity_bonus
        * directional_consistency
    )

    valid_core = core_score[valid]
    valid_core = valid_core[
        np.isfinite(valid_core) & (valid_core > 0.0)
    ]

    if valid_core.size:
        exploration_floor = 0.012 * float(np.mean(valid_core))
    else:
        exploration_floor = 0.0

    score = core_score + exploration_floor * valid.astype(float)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    positive_scores = score[valid]
    positive_scores = positive_scores[
        np.isfinite(positive_scores) & (positive_scores > 0.0)
    ]

    if positive_scores.size:
        normalization = float(np.mean(positive_scores))
        if np.isfinite(normalization) and normalization > 0.0:
            score /= normalization

    return score
