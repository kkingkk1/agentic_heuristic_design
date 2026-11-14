import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally scaled edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite & off_diagonal

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps

    reverse_valid = valid.T
    both_valid = valid & reverse_valid

    symmetric_distance = np.full((n, n), np.inf, dtype=float)
    symmetric_distance[both_valid] = (
        d[both_valid] + d.T[both_valid]
    ) * 0.5

    only_forward = valid & ~reverse_valid
    symmetric_distance[only_forward] = d[only_forward]

    only_reverse = reverse_valid & ~valid
    symmetric_distance[only_reverse] = d.T[only_reverse]

    np.fill_diagonal(symmetric_distance, np.inf)

    finite_sym = np.isfinite(symmetric_distance)
    all_values = symmetric_distance[finite_sym]

    if all_values.size == 0:
        return result

    global_scale = float(np.median(all_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        positive_values = all_values[all_values > 0.0]
        if positive_values.size:
            global_scale = float(np.median(positive_values))
        else:
            global_scale = 1.0
    global_scale = max(global_scale, eps)

    nearest_count = min(4, max(1, n - 1))
    local_scale = np.empty(n, dtype=float)
    local_regret = np.zeros(n, dtype=float)
    rank_matrix = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        row = symmetric_distance[i]
        finite_indices = np.flatnonzero(np.isfinite(row))

        if finite_indices.size == 0:
            local_scale[i] = global_scale
            continue

        ordered = finite_indices[
            np.argsort(row[finite_indices], kind="stable")
        ]
        distances = row[ordered]
        take = distances[:nearest_count]

        local_scale[i] = float(np.median(take))
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
    local_scale = 0.7 * local_scale + 0.3 * global_scale
    local_scale = np.maximum(local_scale, eps)

    finite_rank = np.isfinite(rank_matrix)
    adaptive_rank_scale = max(1.0, np.sqrt(float(n)))

    rank_term = np.exp(
        -(
            rank_matrix
            + rank_matrix.T
        )
        / adaptive_rank_scale
    )
    rank_term = np.where(finite_rank & finite_rank.T, rank_term, 0.0)

    neighborhood_size = min(4, max(1, n - 1))
    mutual_neighbor = (
        (rank_matrix < neighborhood_size)
        & (rank_matrix.T < neighborhood_size)
    )

    mutual_bonus = 1.0 + 0.85 * mutual_neighbor.astype(float)
    rank_bonus = 0.35 + 0.65 * rank_term

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid,
    )

    distance_term = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    scarcity = 1.0 + 0.30 * (
        local_regret[:, None] + local_regret[None, :]
    )

    global_quality = np.exp(
        -np.clip(
            np.divide(
                symmetric_distance,
                global_scale,
                out=np.full_like(symmetric_distance, np.inf),
                where=finite_sym,
            ),
            0.0,
            60.0,
        )
    )
    global_quality = np.nan_to_num(
        global_quality,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    score = (
        distance_term
        * rank_bonus
        * mutual_bonus
        * scarcity
        * (0.75 + 0.25 * global_quality)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    valid_scores = score[valid]
    positive_scores = valid_scores[valid_scores > 0.0]

    if positive_scores.size:
        normalization = float(np.mean(positive_scores))
        if np.isfinite(normalization) and normalization > 0.0:
            score /= normalization

    return score
