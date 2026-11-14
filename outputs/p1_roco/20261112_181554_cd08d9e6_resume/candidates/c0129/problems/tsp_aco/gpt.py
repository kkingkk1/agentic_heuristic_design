import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally normalized, symmetric TSP edge-priority indicators."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    finite_nonnegative = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite_nonnegative, False)

    # Use only mutually valid directed entries so the returned heuristic is symmetric.
    valid = finite_nonnegative & finite_nonnegative.T
    if not np.any(valid):
        return result

    sym_d = np.zeros_like(d, dtype=float)
    sym_d[valid] = 0.5 * (d[valid] + d.T[valid])

    inf = np.inf
    work = np.where(valid, sym_d, inf)
    sorted_distances = np.sort(work, axis=1)

    finite_values = sym_d[valid]
    global_scale = float(np.median(finite_values))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        positive_values = finite_values[finite_values > 0.0]
        global_scale = (
            float(np.median(positive_values))
            if positive_values.size
            else 1.0
        )
    global_scale = max(global_scale, np.finfo(float).eps)

    k = min(5, n - 1)
    local_scale = np.empty(n, dtype=float)
    best = np.full(n, inf, dtype=float)
    second = np.full(n, inf, dtype=float)

    for i in range(n):
        row_values = sorted_distances[i]
        finite_row = row_values[np.isfinite(row_values)]

        if finite_row.size:
            best[i] = finite_row[0]
            if finite_row.size > 1:
                second[i] = finite_row[1]

            local_scale[i] = max(
                float(np.median(finite_row[:k])),
                np.finfo(float).eps,
            )
        else:
            local_scale[i] = global_scale

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    safe_best = np.where(np.isfinite(best), best, local_scale)
    safe_second = np.where(
        np.isfinite(second),
        second,
        safe_best * 2.0,
    )

    regret = np.clip(
        (safe_second - safe_best)
        / np.maximum(safe_second, np.finfo(float).eps),
        0.0,
        1.0,
    )

    urgency = np.sqrt(
        (1.0 + 0.9 * regret[:, None])
        * (1.0 + 0.9 * regret[None, :])
    )

    normalized_local = np.divide(
        sym_d,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full_like(sym_d, inf),
        where=valid,
    )
    local_quality = np.exp(-np.clip(normalized_local, 0.0, 60.0))

    # Retain absolute edge strength in addition to local comparisons.
    absolute_quality = np.exp(
        -np.clip(sym_d / global_scale, 0.0, 60.0)
    )

    row_ratio = np.divide(
        sym_d,
        local_scale[:, None],
        out=np.full_like(sym_d, inf),
        where=valid,
    )
    col_ratio = np.divide(
        sym_d,
        local_scale[None, :],
        out=np.full_like(sym_d, inf),
        where=valid,
    )

    row_rank = np.argsort(
        np.argsort(np.where(valid, row_ratio, inf), axis=1),
        axis=1,
    )
    col_rank = np.argsort(
        np.argsort(np.where(valid, col_ratio, inf), axis=0),
        axis=0,
    )

    rank_quality = 1.0 / np.sqrt(
        1.0 + row_rank.astype(float) + col_rank.astype(float)
    )

    nearest = valid & (
        sym_d <= safe_best[:, None] + 1e-12 * np.maximum(global_scale, 1.0)
    )
    mutual_nearest = nearest & nearest.T
    one_sided_nearest = nearest | nearest.T

    nearest_bonus = (
        1.0
        + 0.35 * mutual_nearest.astype(float)
        + 0.10 * one_sided_nearest.astype(float)
    )

    # Build compact local neighborhoods for a bridge/diversity signal.
    neighborhood = np.zeros((n, n), dtype=float)
    for i in range(n):
        order = np.argsort(np.where(valid[i], sym_d[i], inf))
        chosen = order[:k]
        chosen = chosen[np.isfinite(sym_d[i, chosen])]
        neighborhood[i, chosen] = 1.0

    neighborhood_sizes = np.maximum(neighborhood.sum(axis=1), 1.0)
    overlap = np.divide(
        neighborhood @ neighborhood.T,
        np.minimum(neighborhood_sizes[:, None], neighborhood_sizes[None, :]),
        out=np.zeros_like(sym_d),
        where=np.minimum(
            neighborhood_sizes[:, None],
            neighborhood_sizes[None, :],
        )
        > 0,
    )

    # Prefer useful inter-neighborhood bridges without discarding local edges.
    bridge_factor = 1.0 + 0.22 * (1.0 - np.clip(overlap, 0.0, 1.0))

    # Preserve long-term options: edges whose endpoints retain alternatives
    # receive a mild bonus, while endpoint regret still highlights scarce links.
    reserve = np.minimum(
        safe_second / np.maximum(local_scale, np.finfo(float).eps),
        4.0,
    )
    reserve_factor = 1.0 + 0.12 * np.tanh(
        0.5 * (reserve[:, None] + reserve[None, :])
    )

    score = (
        local_quality
        * absolute_quality
        * rank_quality
        * urgency
        * nearest_bonus
        * bridge_factor
        * reserve_factor
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = 0.5 * (score + score.T)

    total = float(score.sum())
    if not np.isfinite(total) or total <= 0.0:
        result[valid] = 1.0 / float(np.count_nonzero(valid))
        return result

    # A small global floor prevents early sampling collapse while preserving
    # the absolute ordering and strength of the learned priorities.
    valid_count = float(np.count_nonzero(valid))
    diversity_floor = 0.025 / valid_count
    result = 0.975 * (score / total)
    result[valid] += diversity_floor
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    return result
