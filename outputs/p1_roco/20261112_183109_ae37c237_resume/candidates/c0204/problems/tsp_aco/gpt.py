import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    score = np.zeros_like(d, dtype=float)

    if n < 2:
        return score

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return score

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    ordered_rows = np.sort(work, axis=1)
    ordered_cols = np.sort(work, axis=0)

    best_out = ordered_rows[:, 0]
    best_in = ordered_cols[0, :]

    second_out = (
        ordered_rows[:, 1] if n > 2 else best_out.copy()
    )
    second_in = (
        ordered_cols[1, :] if n > 2 else best_in.copy()
    )

    safe_best_out = np.where(np.isfinite(best_out), best_out, 1.0)
    safe_best_in = np.where(np.isfinite(best_in), best_in, 1.0)
    safe_second_out = np.where(
        np.isfinite(second_out),
        second_out,
        2.0 * safe_best_out,
    )
    safe_second_in = np.where(
        np.isfinite(second_in),
        second_in,
        2.0 * safe_best_in,
    )

    out_scale = np.maximum(
        0.5 * (safe_best_out + safe_second_out),
        eps,
    )
    in_scale = np.maximum(
        0.5 * (safe_best_in + safe_second_in),
        eps,
    )

    out_regret = np.clip(
        (safe_second_out - safe_best_out)
        / np.maximum(safe_second_out, eps),
        0.0,
        1.0,
    )
    in_regret = np.clip(
        (safe_second_in - safe_best_in)
        / np.maximum(safe_second_in, eps),
        0.0,
        1.0,
    )

    pair_scale = np.sqrt(out_scale[:, None] * in_scale[None, :])

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid & (pair_scale > 0.0),
    )

    distance_term = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    out_ratio = np.divide(
        d,
        out_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    in_ratio = np.divide(
        d,
        in_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    out_quality = np.exp(-np.clip(out_ratio, 0.0, 60.0))
    in_quality = np.exp(-np.clip(in_ratio, 0.0, 60.0))
    mutual_quality = np.sqrt(out_quality * in_quality)

    row_order = np.argsort(work, axis=1, kind="stable")
    row_ranks = np.full((n, n), np.inf, dtype=float)
    row_ranks[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_ranks = np.full((n, n), np.inf, dtype=float)
    col_ranks[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    finite_row_ranks = np.where(valid, row_ranks, np.inf)
    finite_col_ranks = np.where(valid, col_ranks, np.inf)

    rank_scale = max(1.0, np.sqrt(float(n)))
    mutual_neighbor = np.exp(
        -0.5 * (finite_row_ranks + finite_col_ranks)
        / rank_scale
    )
    mutual_neighbor = np.nan_to_num(
        mutual_neighbor,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    competitive_threshold_out = 1.75 * safe_best_out
    competitive_threshold_in = 1.75 * safe_best_in

    competitive_out = (
        valid
        & (d <= competitive_threshold_out[:, None])
    )
    competitive_in = (
        valid
        & (d <= competitive_threshold_in[None, :])
    )

    out_count = competitive_out.sum(axis=1).astype(float)
    in_count = competitive_in.sum(axis=0).astype(float)

    max_count = max(1.0, float(n - 1))
    out_scarcity = 1.0 - np.minimum(out_count / max_count, 1.0)
    in_scarcity = 1.0 - np.minimum(in_count / max_count, 1.0)

    scarcity_bonus = 1.0 + 0.08 * np.sqrt(
        out_scarcity[:, None] * in_scarcity[None, :]
    )

    continuation_out = np.divide(
        d,
        out_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    continuation_in = np.divide(
        d,
        in_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    continuation_out_quality = np.exp(
        -np.clip(continuation_out, 0.0, 60.0)
    )
    continuation_in_quality = np.exp(
        -np.clip(continuation_in, 0.0, 60.0)
    )

    best_after_j = np.max(
        np.where(valid, continuation_out_quality, 0.0),
        axis=1,
    )
    best_before_i = np.max(
        np.where(valid, continuation_in_quality, 0.0),
        axis=0,
    )

    continuation_potential = np.sqrt(
        best_after_j[None, :] * best_before_i[:, None]
    )

    continuation_bonus = 1.0 + 0.10 * np.clip(
        continuation_potential,
        0.0,
        1.0,
    )

    regret_bonus = 1.0 + 0.32 * (
        out_regret[:, None] + in_regret[None, :]
    )

    long_edge_penalty = np.exp(
        -0.22 * np.maximum(relative_distance - 1.0, 0.0)
    )

    score = (
        distance_term
        * (0.62 + 0.23 * mutual_quality + 0.15 * mutual_neighbor)
        * regret_bonus
        * scarcity_bonus
        * continuation_bonus
        * long_edge_penalty
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    row_totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        row_totals,
        out=np.zeros_like(score),
        where=row_totals > 0.0,
    )
