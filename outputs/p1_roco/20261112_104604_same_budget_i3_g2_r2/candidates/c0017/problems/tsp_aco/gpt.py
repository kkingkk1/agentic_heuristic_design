import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be a square matrix")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    valid = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    np.fill_diagonal(valid, False)

    score = np.zeros((n, n), dtype=float)
    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    distances = distance_matrix.copy()

    positive = distances[valid & (distances > 0.0)]
    if positive.size:
        distance_floor = max(float(np.quantile(positive, 0.05)), eps)
    else:
        distance_floor = eps

    safe_distance = np.where(
        valid,
        np.maximum(distances, distance_floor),
        np.nan,
    )

    def normalize_valid(values: np.ndarray) -> np.ndarray:
        values = np.where(valid, np.maximum(values, 0.0), 0.0)
        selected = values[valid]

        if selected.size == 0:
            return np.zeros_like(values, dtype=float)

        low = float(np.quantile(selected, 0.05))
        high = float(np.quantile(selected, 0.95))

        if high - low <= eps:
            return np.where(valid, 1.0, 0.0)

        result = np.zeros_like(values, dtype=float)
        clipped = np.clip(values, low, high)
        result[valid] = (clipped[valid] - low) / (high - low + eps)
        return result

    def normalize_rows(values: np.ndarray) -> np.ndarray:
        values = np.where(valid, np.maximum(values, 0.0), 0.0)
        row_max = np.max(values, axis=1, keepdims=True)
        return np.divide(
            values,
            row_max,
            out=np.zeros_like(values, dtype=float),
            where=row_max > eps,
        )

    log_distance = np.where(
        valid,
        np.log1p(safe_distance / distance_floor),
        np.nan,
    )

    global_closeness = normalize_valid(
        -np.nan_to_num(log_distance, nan=0.0)
    )

    zero_bonus = np.where(
        valid & (distances <= eps),
        1.0,
        0.0,
    )
    global_closeness = np.maximum(global_closeness, zero_bonus)

    row_order = np.argsort(
        np.where(valid, safe_distance, np.inf),
        axis=1,
        kind="mergesort",
    )
    row_rank = np.full((n, n), n + 1.0, dtype=float)

    for i in range(n):
        indices = row_order[i][valid[i, row_order[i]]]
        row_rank[i, indices] = np.arange(1, indices.size + 1)

    column_valid = valid.T
    column_order = np.argsort(
        np.where(column_valid, safe_distance.T, np.inf),
        axis=1,
        kind="mergesort",
    )
    column_rank = np.full((n, n), n + 1.0, dtype=float)

    for j in range(n):
        indices = column_order[j][column_valid[j, column_order[j]]]
        column_rank[j, indices] = np.arange(1, indices.size + 1)

    row_count = np.sum(valid, axis=1, keepdims=True)
    column_count = np.sum(column_valid, axis=1, keepdims=True)

    outgoing_quality = np.where(
        valid,
        1.0
        - (row_rank - 1.0) / np.maximum(row_count, 1.0),
        0.0,
    )
    incoming_quality = np.where(
        valid,
        1.0
        - (column_rank.T - 1.0)
        / np.maximum(column_count.T, 1.0),
        0.0,
    )

    endpoint_agreement = np.sqrt(
        np.maximum(outgoing_quality, 0.0)
        * np.maximum(incoming_quality, 0.0)
    )

    local_signal = normalize_rows(
        0.65 * outgoing_quality
        + 0.35 * endpoint_agreement
    )

    global_mass = np.where(valid, global_closeness, 0.0)
    row_mass = np.sum(global_mass, axis=1, keepdims=True)
    column_mass = np.sum(global_mass, axis=0, keepdims=True)

    row_share = np.divide(
        global_mass,
        row_mass,
        out=np.zeros_like(global_mass),
        where=row_mass > eps,
    )
    column_share = np.divide(
        global_mass,
        column_mass,
        out=np.zeros_like(global_mass),
        where=column_mass > eps,
    )

    row_dominance = np.minimum(n * row_share, 1.0)
    column_dominance = np.minimum(n * column_share, 1.0)

    diversity_reserve = 1.0 - np.sqrt(
        row_dominance * column_dominance
    )
    diversity_reserve = np.where(valid, diversity_reserve, 0.0)

    global_diversity = normalize_valid(
        np.sqrt(
            np.maximum(global_closeness, 0.0)
            * (0.45 + 0.55 * diversity_reserve)
        )
    )

    row_sorted = np.sort(
        np.where(valid, safe_distance, np.inf),
        axis=1,
    )
    column_sorted = np.sort(
        np.where(column_valid, safe_distance.T, np.inf),
        axis=1,
    )

    row_best = row_sorted[:, 0:1]
    row_second = row_sorted[:, 1:2] if n > 1 else row_best
    column_best = column_sorted[:, 0:1]
    column_second = column_sorted[:, 1:2] if n > 1 else column_best

    row_regret = np.divide(
        np.maximum(row_second - row_best, 0.0),
        row_second + eps,
        out=np.zeros_like(row_best),
        where=np.isfinite(row_second),
    )
    column_regret = np.divide(
        np.maximum(column_second - column_best, 0.0),
        column_second + eps,
        out=np.zeros_like(column_best),
        where=np.isfinite(column_second),
    )

    row_opportunity = np.clip(
        0.50 + 0.50 * row_regret,
        0.0,
        1.0,
    )
    column_opportunity = np.clip(
        0.50 + 0.50 * column_regret,
        0.0,
        1.0,
    )

    endpoint_scarcity = np.sqrt(
        row_opportunity * column_opportunity.T
    )

    two_step = np.full((n, n), np.nan, dtype=float)

    if n > 2:
        for i in range(n):
            for j in range(n):
                if not valid[i, j]:
                    continue

                intermediates = valid[i, :] & valid[:, j]
                intermediates[i] = False
                intermediates[j] = False

                if np.any(intermediates):
                    candidates = (
                        safe_distance[i, intermediates]
                        + safe_distance[intermediates, j]
                    )
                    two_step[i, j] = np.min(candidates)

    direct_advantage = np.full((n, n), 0.5, dtype=float)
    finite_two_step = valid & np.isfinite(two_step)

    direct_advantage[finite_two_step] = np.divide(
        two_step[finite_two_step],
        safe_distance[finite_two_step]
        + two_step[finite_two_step]
        + eps,
    )

    direct_advantage = np.clip(direct_advantage, 0.0, 1.0)
    direct_advantage = np.where(valid, direct_advantage, 0.0)

    replacement_scarcity = normalize_valid(
        0.65 * endpoint_scarcity
        + 0.35 * direct_advantage
    )

    long_term_potential = normalize_valid(
        0.55 * endpoint_scarcity
        + 0.25 * direct_advantage
        + 0.20 * diversity_reserve
    )

    score = (
        0.28 * global_diversity
        + 0.24 * local_signal
        + 0.30 * long_term_potential
        + 0.10 * replacement_scarcity
        + 0.08 * endpoint_agreement
    )

    score = np.where(valid, np.maximum(score, 0.0), 0.0)
    np.fill_diagonal(score, 0.0)

    return score.astype(float, copy=False)
