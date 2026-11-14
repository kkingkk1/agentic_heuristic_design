import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative TSP edge-priority indicators with mild look-ahead."""

    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    valid = finite & ~np.eye(n, dtype=bool)

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

    positive_values = all_values[all_values > eps]
    scale_values = positive_values if positive_values.size else all_values

    global_scale = float(np.median(scale_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    degree = max(1, n - 1)
    nearest_count = min(4, degree)

    local_scale = np.full(n, global_scale, dtype=float)
    local_regret = np.zeros(n, dtype=float)
    rank_matrix = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        row = symmetric_distance[i]
        indices = np.flatnonzero(np.isfinite(row))

        if indices.size == 0:
            continue

        ordered = indices[np.argsort(row[indices], kind="stable")]
        distances = row[ordered]

        local_scale[i] = float(np.median(distances[:nearest_count]))

        if distances.size >= 2:
            best = float(distances[0])
            second = float(distances[1])
            denominator = max(second, global_scale, eps)
            local_regret[i] = np.clip(
                (second - best) / denominator,
                0.0,
                1.0,
            )

        rank_matrix[i, ordered] = np.arange(ordered.size, dtype=float)

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(
        0.75 * local_scale + 0.25 * global_scale,
        eps,
    )

    finite_rank = np.isfinite(rank_matrix)
    rank_scale = max(1.0, np.sqrt(float(n)))

    rank_term = np.exp(
        -(rank_matrix + rank_matrix.T) / rank_scale
    )
    rank_term = np.where(
        finite_rank & finite_rank.T,
        rank_term,
        0.0,
    )

    rank_bonus = 0.40 + 0.60 * rank_term

    mutual_neighbor = (
        (rank_matrix < nearest_count)
        & (rank_matrix.T < nearest_count)
    )
    mutual_bonus = 1.0 + 0.65 * mutual_neighbor.astype(float)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        symmetric_distance,
        pair_scale,
        out=np.full_like(symmetric_distance, np.inf),
        where=finite_sym,
    )

    distance_term = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
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

    scarcity_bonus = 1.0 + 0.22 * (
        local_regret[:, None] + local_regret[None, :]
    )

    top_two = (
        (rank_matrix <= 1.0)
        & (rank_matrix.T <= 1.0)
    )
    lookahead_bonus = 1.0 + 0.16 * top_two.astype(float)

    core_score = (
        distance_term
        * rank_bonus
        * mutual_bonus
        * scarcity_bonus
        * lookahead_bonus
        * (0.82 + 0.18 * global_quality)
    )
    core_score[~valid] = 0.0

    # Multi-candidate continuation potential.
    # It uses a soft aggregate of several good neighbors instead of
    # relying only on the single best continuation.
    continuation_temperature = max(
        0.55,
        0.85 * np.sqrt(float(n)),
    )

    rank_weights = np.zeros_like(rank_matrix)
    rank_weights[finite_rank] = np.exp(
        -rank_matrix[finite_rank] / continuation_temperature
    )

    row_mass = np.sum(rank_weights, axis=1)
    row_potential = np.divide(
        np.sum(rank_weights * core_score, axis=1),
        row_mass,
        out=np.zeros(n, dtype=float),
        where=row_mass > eps,
    )

    col_mass = np.sum(rank_weights, axis=0)
    col_potential = np.divide(
        np.sum(rank_weights * core_score, axis=0),
        col_mass,
        out=np.zeros(n, dtype=float),
        where=col_mass > eps,
    )

    potential_scale = float(np.max(core_score[valid]))
    if not np.isfinite(potential_scale) or potential_scale <= eps:
        potential_scale = 1.0

    outgoing_potential = np.clip(
        row_potential / potential_scale,
        0.0,
        1.0,
    )
    incoming_potential = np.clip(
        col_potential / potential_scale,
        0.0,
        1.0,
    )

    continuation = np.sqrt(
        outgoing_potential[None, :] * incoming_potential[:, None]
    )
    continuation = np.clip(continuation, 0.0, 1.0)

    score = core_score * (0.90 + 0.10 * continuation)

    # Mild endpoint coverage correction.
    row_mass = score.sum(axis=1)
    col_mass = score.sum(axis=0)

    row_support = np.sqrt(np.maximum(row_mass, eps))
    col_support = np.sqrt(np.maximum(col_mass, eps))

    diversity_factor = 1.0 / np.sqrt(
        row_support[:, None] * col_support[None, :]
    )

    valid_diversity = diversity_factor[valid]
    if valid_diversity.size:
        diversity_factor /= max(
            float(np.mean(valid_diversity)),
            eps,
        )

    score *= 0.94 + 0.06 * np.clip(
        diversity_factor,
        0.5,
        2.0,
    )

    # Keep a small quality-aware floor for valid but under-ranked edges.
    valid_global_quality = global_quality[valid]
    exploration_floor = (
        0.02 * float(np.mean(valid_global_quality))
        if valid_global_quality.size
        else 0.0
    )

    score += exploration_floor * valid.astype(float)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    positive_scores = score[valid]
    positive_scores = positive_scores[positive_scores > 0.0]

    if positive_scores.size:
        normalization = float(np.mean(positive_scores))
        if np.isfinite(normalization) and normalization > 0.0:
            score /= normalization

    return score
