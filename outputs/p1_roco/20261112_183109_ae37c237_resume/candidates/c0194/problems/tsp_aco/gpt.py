import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative but mildly exploratory TSP edge indicators."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros((n, n), dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    reverse_valid = valid.T
    both_valid = valid & reverse_valid

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])
    symmetric[valid & ~reverse_valid] = d[valid & ~reverse_valid]
    symmetric[reverse_valid & ~valid] = d.T[reverse_valid & ~valid]
    np.fill_diagonal(symmetric, np.inf)

    finite = np.isfinite(symmetric)
    values = symmetric[finite]
    if values.size == 0:
        return score

    eps = np.finfo(float).eps
    positive = values[values > eps]
    global_scale = float(np.median(positive if positive.size else values))
    global_scale = max(global_scale, eps)

    degree = max(1, n - 1)
    local_k = min(degree, max(2, int(np.ceil(np.sqrt(float(n))))))
    ordered = np.sort(symmetric, axis=1)

    local_neighbors = ordered[:, :local_k]
    local_scale = np.nanmedian(
        np.where(np.isfinite(local_neighbors), local_neighbors, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(
        0.8 * local_scale + 0.2 * global_scale,
        eps,
    )

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    relative = np.divide(
        symmetric,
        np.maximum(pair_scale, eps),
        out=np.full_like(symmetric, np.inf),
        where=finite,
    )

    local_quality = np.exp(-np.clip(relative, 0.0, 60.0))
    local_quality[~valid] = 0.0

    global_quality = np.exp(
        -np.clip(symmetric / global_scale, 0.0, 60.0)
    )
    global_quality = np.nan_to_num(
        global_quality,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    global_quality[~valid] = 0.0

    rank_order = np.argsort(symmetric, axis=1, kind="stable")
    rank = np.full((n, n), np.inf, dtype=float)
    rank[np.arange(n)[:, None], rank_order] = np.arange(n)

    rank_temperature = max(1.0, np.sqrt(float(n)))
    outgoing_rank = np.exp(-rank / rank_temperature)
    outgoing_rank[~valid] = 0.0

    incoming_rank = outgoing_rank.T
    mutual_rank = np.sqrt(outgoing_rank * incoming_rank)

    mutual_k = min(degree, max(2, int(np.ceil(np.log2(n + 1.0)))))
    mutual = (
        valid
        & (rank < mutual_k)
        & (rank.T < mutual_k)
    )

    rank_term = 0.42 + 0.58 * mutual_rank
    mutual_term = 1.0 + 0.42 * mutual.astype(float)

    candidate_k = min(
        degree,
        max(2, int(np.ceil(1.25 * np.sqrt(float(n))))),
    )
    top_quality = np.sort(local_quality, axis=1)[:, ::-1][:, :candidate_k]
    top_quality = np.where(np.isfinite(top_quality), top_quality, 0.0)

    candidate_mass = top_quality.sum(axis=1)
    support = np.count_nonzero(top_quality > 0.0, axis=1)
    support_factor = np.minimum(
        support / float(candidate_k),
        1.0,
    )

    continuation_out = (
        0.72 * candidate_mass / max(float(candidate_k), 1.0)
        + 0.28 * support_factor
    )
    continuation_out = np.clip(continuation_out, 0.0, 1.0)

    continuation_in = continuation_out.copy()
    continuation = np.sqrt(
        continuation_out[None, :] * continuation_in[:, None]
    )

    second = ordered[:, min(1, degree - 1)]
    nearest = ordered[:, 0]
    second = np.where(np.isfinite(second), second, global_scale)
    nearest = np.where(np.isfinite(nearest), nearest, global_scale)

    regret = np.clip(
        (second - nearest) / np.maximum(second, eps),
        0.0,
        1.0,
    )

    score = (
        local_quality
        * rank_term
        * mutual_term
        * (0.90 + 0.10 * global_quality)
    )

    # Bounded lookahead: useful for exploration, but weaker than local quality.
    score *= 0.92 + 0.10 * continuation

    # Small scarcity signal for nodes with a clear nearest-neighbor advantage.
    score *= 1.0 + 0.08 * (
        regret[:, None] + regret[None, :]
    )

    # Weak global floor prevents complete starvation of valid edges.
    mean_global_quality = float(
        np.mean(global_quality[valid])
    )
    score += 0.006 * mean_global_quality * valid.astype(float)

    # Mild capacity correction raises underrepresented rows and columns.
    row_mass = score.sum(axis=1)
    col_mass = score.sum(axis=0)

    positive_rows = row_mass[row_mass > 0.0]
    positive_cols = col_mass[col_mass > 0.0]
    row_reference = (
        float(np.median(positive_rows))
        if positive_rows.size
        else 1.0
    )
    col_reference = (
        float(np.median(positive_cols))
        if positive_cols.size
        else 1.0
    )

    row_need = np.sqrt(row_reference / np.maximum(row_mass, eps))
    col_need = np.sqrt(col_reference / np.maximum(col_mass, eps))
    row_need = np.clip(row_need, 0.82, 1.28)
    col_need = np.clip(col_need, 0.82, 1.28)

    capacity = np.sqrt(row_need[:, None] * col_need[None, :])
    capacity /= max(float(np.mean(capacity[valid])), eps)

    score *= 0.96 + 0.04 * np.clip(capacity, 0.70, 1.40)
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

    if not positive.size:
        score[valid] = 1.0
        return score

    mean_score = float(np.mean(positive))
    if not np.isfinite(mean_score) or mean_score <= 0.0:
        score[valid] = 1.0
        return score

    return score / mean_score
