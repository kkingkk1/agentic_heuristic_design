import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust, direction-aware edge priorities for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    eps = np.finfo(float).eps
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d >= 0.0) & off_diagonal

    if not np.any(valid):
        return result

    positive = d[valid & (d > eps)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = d[valid]
        global_scale = (
            float(np.median(finite_values))
            if finite_values.size
            else 1.0
        )

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    degree = n - 1
    local_k = max(1, min(degree, int(np.ceil(np.sqrt(float(n))))))

    work = np.where(valid, d, np.inf)

    outgoing_scale = np.full(n, global_scale, dtype=float)
    incoming_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row_values = np.sort(work[i])
        row_values = row_values[np.isfinite(row_values)]
        if row_values.size:
            selected = row_values[:local_k]
            positive_selected = selected[selected > eps]
            outgoing_scale[i] = (
                float(np.median(positive_selected))
                if positive_selected.size
                else max(float(np.median(selected)), global_scale)
            )

        column_values = np.sort(work[:, i])
        column_values = column_values[np.isfinite(column_values)]
        if column_values.size:
            selected = column_values[:local_k]
            positive_selected = selected[selected > eps]
            incoming_scale[i] = (
                float(np.median(positive_selected))
                if positive_selected.size
                else max(float(np.median(selected)), global_scale)
            )

    outgoing_scale = np.maximum(
        0.80 * outgoing_scale + 0.20 * global_scale,
        eps,
    )
    incoming_scale = np.maximum(
        0.80 * incoming_scale + 0.20 * global_scale,
        eps,
    )

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )
    pair_scale = np.maximum(
        0.75 * pair_scale + 0.25 * global_scale,
        eps,
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    distance_term = np.exp(
        -np.clip(
            0.78 * relative_distance
            + 0.22 * np.divide(
                d,
                global_scale,
                out=np.full_like(d, np.inf, dtype=float),
                where=valid,
            ),
            0.0,
            60.0,
        )
    )
    distance_term[~valid] = 0.0

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

    row_rank[~valid] = np.inf
    col_rank[~valid] = np.inf

    rank_temperature = max(1.0, 0.90 * np.sqrt(float(n)))
    rank_quality = np.sqrt(
        np.exp(-np.clip(row_rank, 0.0, 60.0) / rank_temperature)
        * np.exp(-np.clip(col_rank, 0.0, 60.0) / rank_temperature)
    )
    rank_quality[~valid] = 0.0

    mutual_limit = max(1, min(local_k, int(np.ceil(np.log2(n + 1.0)))))
    mutual = (
        valid
        & (row_rank < mutual_limit)
        & (col_rank < mutual_limit)
    )

    rank_term = 0.52 + 0.48 * rank_quality
    mutual_term = 1.0 + 0.24 * mutual.astype(float)

    reciprocal = valid & valid.T
    reverse_distance = np.where(reciprocal, d.T, d)
    direction_ratio = np.divide(
        d,
        np.maximum(reverse_distance, eps),
        out=np.ones_like(d, dtype=float),
        where=valid,
    )
    direction_term = np.exp(
        -0.14 * np.clip(direction_ratio - 1.0, 0.0, 20.0)
    )
    direction_term[~valid] = 0.0

    core = (
        distance_term
        * rank_term
        * mutual_term
        * direction_term
    )
    core[~valid] = 0.0
    core = np.nan_to_num(
        core,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    candidate_k = max(
        1,
        min(degree, int(np.ceil(1.5 * np.sqrt(float(n))))),
    )
    candidate_temperature = max(
        0.06,
        0.20 / np.sqrt(float(n)),
    )

    def candidate_potential(values: np.ndarray) -> np.ndarray:
        potential = np.zeros(n, dtype=float)

        for node in range(n):
            candidates = values[node]
            candidates = candidates[candidates > 0.0]

            if candidates.size == 0:
                continue

            candidates = np.sort(candidates)[::-1][:candidate_k]
            peak = float(candidates[0])

            weights = np.exp(
                np.clip(
                    (candidates - peak) / candidate_temperature,
                    -50.0,
                    0.0,
                )
            )
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                continue

            weights /= weight_sum
            weighted_quality = float(np.sum(weights * candidates))
            support = min(
                candidates.size / float(candidate_k),
                1.0,
            )

            entropy = -float(
                np.sum(weights * np.log(np.maximum(weights, eps)))
            )
            entropy /= max(np.log(float(candidate_k) + 1.0), 1.0)

            potential[node] = weighted_quality * support * (
                0.75 + 0.25 * entropy
            )

        return np.clip(potential, 0.0, 1.0)

    outgoing_potential = candidate_potential(core)
    incoming_potential = candidate_potential(core.T)

    continuation = np.sqrt(
        incoming_potential[:, None]
        * outgoing_potential[None, :]
    )
    continuation = np.clip(continuation, 0.0, 1.0)

    score = core * (0.88 + 0.12 * continuation)

    row_mass = score.sum(axis=1)
    col_mass = score.sum(axis=0)

    positive_rows = row_mass[row_mass > 0.0]
    positive_cols = col_mass[col_mass > 0.0]

    reference_row = (
        float(np.median(positive_rows))
        if positive_rows.size
        else 1.0
    )
    reference_col = (
        float(np.median(positive_cols))
        if positive_cols.size
        else 1.0
    )

    row_need = np.clip(
        np.sqrt(reference_row / np.maximum(row_mass, eps)),
        0.80,
        1.40,
    )
    col_need = np.clip(
        np.sqrt(reference_col / np.maximum(col_mass, eps)),
        0.80,
        1.40,
    )

    coverage = np.sqrt(
        row_need[:, None] * col_need[None, :]
    )
    score *= 0.94 + 0.06 * np.clip(coverage, 0.70, 1.50)

    valid_scores = score[valid]
    valid_scores = valid_scores[
        np.isfinite(valid_scores) & (valid_scores > 0.0)
    ]

    if valid_scores.size:
        floor = 0.012 * float(np.mean(valid_scores))
        exploration = np.sqrt(
            row_need[:, None] * col_need[None, :]
        )
        score += floor * exploration * valid.astype(float)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    positive = score[valid]
    positive = positive[
        np.isfinite(positive) & (positive > 0.0)
    ]

    if not positive.size:
        result[valid] = 1.0
        return result

    normalization = float(np.mean(positive))
    if not np.isfinite(normalization) or normalization <= 0.0:
        result[valid] = 1.0
        return result

    result = score / normalization
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)
    return result
