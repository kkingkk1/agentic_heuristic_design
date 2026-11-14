import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally calibrated edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    # Use symmetric distances for robust neighborhood statistics while
    # retaining the original directed edge values in the final score.
    reverse_valid = valid & valid.T
    symmetric = np.where(
        reverse_valid,
        0.5 * (d + d.T),
        np.where(valid, d, np.where(valid.T, d.T, np.inf)),
    )
    symmetric[~(valid | valid.T)] = np.inf
    np.fill_diagonal(symmetric, np.inf)

    finite_values = symmetric[np.isfinite(symmetric)]
    if finite_values.size == 0:
        return result

    global_scale = float(np.median(finite_values))
    if not np.isfinite(global_scale) or global_scale <= eps:
        positive = finite_values[finite_values > eps]
        global_scale = (
            float(np.median(positive)) if positive.size else 1.0
        )
    global_scale = max(global_scale, eps)

    # Robust local scales from the nearest neighbors.
    ordered = np.sort(symmetric, axis=1)
    k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = ordered[:, :k]
    local_scale = np.nanmedian(
        np.where(np.isfinite(nearest), nearest, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    # Blend local and global scales to avoid overfitting to dense clusters.
    scale = 0.72 * np.sqrt(local_scale[:, None] * local_scale[None, :])
    scale += 0.28 * global_scale
    scale = np.maximum(scale, eps)

    relative = np.divide(
        symmetric,
        scale,
        out=np.full_like(symmetric, np.inf),
        where=np.isfinite(symmetric),
    )

    base_quality = np.exp(-np.clip(relative, 0.0, 50.0))

    # Adaptive ranks at both endpoints.
    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(
        n, dtype=float
    )[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.full((n, n), np.inf, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(
        n, dtype=float
    )[:, None]

    rank_temperature = max(1.0, 0.75 * np.sqrt(float(n)))
    outgoing_rank = np.exp(
        -row_rank / rank_temperature
    )
    incoming_rank = np.exp(
        -col_rank / rank_temperature
    )
    outgoing_rank[~valid] = 0.0
    incoming_rank[~valid] = 0.0

    mutual_support = np.sqrt(outgoing_rank * incoming_rank)

    # Strong bonus for reciprocal-neighbor edges.
    reciprocal_k = max(1, min(n - 1, int(np.ceil(np.log2(n + 1)))))
    mutual_neighbor = (
        (row_rank < reciprocal_k) &
        (col_rank < reciprocal_k) &
        valid
    ).astype(float)

    reciprocal_bonus = 1.0 + 0.85 * mutual_neighbor
    rank_bonus = 0.55 + 0.45 * mutual_support

    # Scarcity: an edge matters more when either endpoint has few
    # comparably good alternatives.
    second = ordered[:, min(1, n - 2)]
    kth = ordered[:, k - 1]

    second = np.where(np.isfinite(second), second, global_scale)
    kth = np.where(np.isfinite(kth), kth, global_scale)

    scarcity = np.clip(
        (kth - local_scale) / np.maximum(kth, eps),
        0.0,
        1.0,
    )
    regret = np.clip(
        (second - local_scale) / np.maximum(second, eps),
        0.0,
        1.0,
    )

    scarcity_bonus = 1.0 + 0.42 * (
        scarcity[:, None] + scarcity[None, :]
    )
    regret_bonus = 1.0 + 0.28 * (
        regret[:, None] + regret[None, :]
    )

    # Long-term continuation potential:
    # the destination should have promising outgoing edges and the source
    # should have promising incoming edges after this edge is selected.
    outgoing_best = np.max(
        np.where(valid, base_quality * outgoing_rank, 0.0),
        axis=1,
    )
    incoming_best = np.max(
        np.where(valid, base_quality * incoming_rank, 0.0),
        axis=0,
    )

    continuation = np.sqrt(
        outgoing_best[:, None] * incoming_best[None, :]
    )
    continuation_bonus = 0.82 + 0.36 * np.clip(continuation, 0.0, 1.0)

    # Penalize edges that are globally long, but retain a nonzero
    # exploratory floor for sparse or irregular geometries.
    long_edge_penalty = np.exp(
        -0.48 * np.maximum(relative - 1.0, 0.0)
    )
    exploration_floor = 0.035

    score = (
        base_quality
        * rank_bonus
        * reciprocal_bonus
        * scarcity_bonus
        * regret_bonus
        * continuation_bonus
        * long_edge_penalty
    )

    score[~valid] = 0.0
    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Mild endpoint balancing prevents global mass from collapsing onto
    # one dense region while preserving globally important edge weights.
    row_mass = score.sum(axis=1)
    col_mass = score.sum(axis=0)

    row_factor = 1.0 / np.sqrt(np.maximum(row_mass, eps))
    col_factor = 1.0 / np.sqrt(np.maximum(col_mass, eps))
    balance = np.sqrt(row_factor[:, None] * col_factor[None, :])
    score *= balance

    valid_count = float(np.count_nonzero(valid))
    uniform = np.zeros_like(score)
    uniform[valid] = 1.0 / max(valid_count, 1.0)

    score = (1.0 - exploration_floor) * score + exploration_floor * uniform
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    total = float(score.sum())
    if not np.isfinite(total) or total <= 0.0:
        result[valid] = 1.0 / max(valid_count, 1.0)
        return result

    return score / total
