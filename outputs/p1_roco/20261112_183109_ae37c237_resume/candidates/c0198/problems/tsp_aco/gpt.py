import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-priority indicators for stochastic TSP solution sampling."""

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
    reverse_valid = valid.T
    both_valid = valid & reverse_valid

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[both_valid] = (
        d[both_valid] + d.T[both_valid]
    ) * 0.5

    only_forward = valid & ~reverse_valid
    symmetric[only_forward] = d[only_forward]

    only_reverse = reverse_valid & ~valid
    symmetric[only_reverse] = d.T[only_reverse]

    np.fill_diagonal(symmetric, np.inf)

    finite_values = symmetric[np.isfinite(symmetric)]
    if finite_values.size == 0:
        return result

    positive_values = finite_values[finite_values > eps]
    scale_values = positive_values if positive_values.size else finite_values

    global_scale = float(np.median(scale_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    degree = max(1, n - 1)
    nearest_count = max(1, min(4, degree))

    ordered = np.argsort(symmetric, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        ordered,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(symmetric, axis=0, kind="stable")
    col_rank = np.full((n, n), np.inf, dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    local_values = np.take_along_axis(
        symmetric,
        ordered[:, :nearest_count],
        axis=1,
    )
    local_scale = np.nanmedian(
        np.where(np.isfinite(local_values), local_values, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(
        0.75 * local_scale + 0.25 * global_scale,
        eps,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    relative_distance = np.divide(
        symmetric,
        pair_scale,
        out=np.full_like(symmetric, np.inf),
        where=np.isfinite(symmetric),
    )
    distance_term = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    global_relative = np.divide(
        symmetric,
        global_scale,
        out=np.full_like(symmetric, np.inf),
        where=np.isfinite(symmetric),
    )
    global_term = np.exp(
        -np.clip(global_relative, 0.0, 60.0)
    )

    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_term = np.exp(
        -(row_rank + col_rank) / rank_scale
    )
    rank_term[~(np.isfinite(row_rank) & np.isfinite(col_rank))] = 0.0

    mutual_neighbor = (
        (row_rank < nearest_count)
        & (col_rank < nearest_count)
    )

    second_index = min(1, degree - 1)
    second_best = np.take_along_axis(
        symmetric,
        ordered[:, second_index:second_index + 1],
        axis=1,
    )[:, 0]
    best = np.take_along_axis(
        symmetric,
        ordered[:, :1],
        axis=1,
    )[:, 0]

    second_best = np.where(
        np.isfinite(second_best),
        second_best,
        global_scale,
    )
    best = np.where(
        np.isfinite(best),
        best,
        global_scale,
    )

    regret = np.divide(
        second_best - best,
        np.maximum(second_best, global_scale),
    )
    regret = np.clip(regret, 0.0, 1.0)

    candidate_k = max(
        1,
        min(degree, int(np.ceil(1.5 * np.sqrt(float(n))))),
    )
    candidate_order = np.argsort(
        np.where(valid, distance_term, -np.inf),
        axis=1,
    )[:, ::-1]
    candidate_indices = candidate_order[:, :candidate_k]
    candidate_values = np.take_along_axis(
        np.where(valid, distance_term, 0.0),
        candidate_indices,
        axis=1,
    )

    candidate_support = np.mean(
        candidate_values > 0.0,
        axis=1,
    )
    candidate_quality = np.mean(
        candidate_values,
        axis=1,
    )
    continuation = 0.65 * candidate_quality + 0.35 * candidate_support

    core_score = (
        distance_term
        * (0.45 + 0.55 * rank_term)
        * (1.0 + 0.38 * mutual_neighbor.astype(float))
        * (
            1.0
            + 0.18 * (regret[:, None] + regret[None, :])
        )
        * (0.84 + 0.16 * global_term)
    )

    continuation_pair = np.sqrt(
        continuation[:, None] * continuation[None, :]
    )
    score = core_score * (
        0.82 + 0.18 * continuation_pair
    )

    valid_global_term = global_term[valid]
    exploration_floor = 0.012 * (
        float(np.mean(valid_global_term))
        if valid_global_term.size
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

    positive = score[valid]
    positive = positive[positive > 0.0]

    if positive.size == 0:
        count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / max(count, 1.0)
        return result

    normalization = float(np.mean(positive))
    if not np.isfinite(normalization) or normalization <= 0.0:
        normalization = 1.0

    return score / normalization
