import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
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

    def endpoint_statistics(matrix: np.ndarray):
        ordered = np.sort(matrix, axis=1)
        best = ordered[:, 0]

        if n > 2:
            second = ordered[:, 1]
        else:
            second = best.copy()

        finite_best = np.isfinite(best)
        safe_best = np.where(finite_best, best, 1.0)
        safe_second = np.where(
            np.isfinite(second),
            second,
            2.0 * safe_best,
        )

        scale = np.maximum(0.5 * (safe_best + safe_second), eps)
        regret = np.clip(
            (safe_second - safe_best)
            / np.maximum(safe_second, eps),
            0.0,
            1.0,
        )
        return scale, regret

    outgoing_scale, outgoing_regret = endpoint_statistics(work)
    incoming_scale, incoming_regret = endpoint_statistics(work.T)

    pair_scale = np.sqrt(
        outgoing_scale[:, None] * incoming_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf),
        where=valid & (pair_scale > 0.0),
    )

    distance_quality = np.exp(
        -np.clip(relative_distance, 0.0, 60.0)
    )

    outgoing_ratio = np.divide(
        d,
        outgoing_scale[:, None],
        out=np.full_like(d, np.inf),
        where=valid,
    )
    incoming_ratio = np.divide(
        d,
        incoming_scale[None, :],
        out=np.full_like(d, np.inf),
        where=valid,
    )

    outgoing_quality = np.exp(
        -np.clip(outgoing_ratio, 0.0, 60.0)
    )
    incoming_quality = np.exp(
        -np.clip(incoming_ratio, 0.0, 60.0)
    )

    ranks_out = np.full((n, n), np.inf, dtype=float)
    row_order = np.argsort(work, axis=1, kind="stable")
    rank_values = np.arange(n, dtype=float)
    ranks_out[np.arange(n)[:, None], row_order] = rank_values[None, :]

    ranks_in = np.full((n, n), np.inf, dtype=float)
    column_order = np.argsort(work, axis=0, kind="stable")
    ranks_in[column_order, np.arange(n)[None, :]] = rank_values[:, None]

    finite_rank_out = np.where(np.isfinite(ranks_out), ranks_out, float(n))
    finite_rank_in = np.where(np.isfinite(ranks_in), ranks_in, float(n))

    rank_scale = max(1.0, np.sqrt(float(n)))
    rank_quality = np.exp(
        -0.5 * (finite_rank_out + finite_rank_in) / rank_scale
    )
    rank_quality = np.where(valid, rank_quality, 0.0)

    nearest_out = np.argmin(work, axis=1)
    nearest_in = np.argmin(work, axis=0)

    mutual_nearest = (
        (np.arange(n)[:, None] == nearest_in[None, :])
        & (np.arange(n)[:, None] * 0 + nearest_out[:, None] == np.arange(n)[None, :])
    )
    mutual_nearest = mutual_nearest.astype(float)

    regret_term = 1.0 + 0.42 * (
        outgoing_regret[:, None] + incoming_regret[None, :]
    )

    best_remaining_out = np.full(n, 0.0, dtype=float)
    best_remaining_in = np.full(n, 0.0, dtype=float)

    for node in range(n):
        outgoing_values = work[node].copy()
        outgoing_values[node] = np.inf
        best_remaining_out[node] = (
            np.min(outgoing_values)
            if np.any(np.isfinite(outgoing_values))
            else 0.0
        )

        incoming_values = work[:, node].copy()
        incoming_values[node] = np.inf
        best_remaining_in[node] = (
            np.min(incoming_values)
            if np.any(np.isfinite(incoming_values))
            else 0.0
        )

    residual_out = np.divide(
        best_remaining_out[None, :],
        np.maximum(outgoing_scale[None, :], eps),
    )
    residual_in = np.divide(
        best_remaining_in[:, None],
        np.maximum(incoming_scale[:, None], eps),
    )

    residual_support = np.exp(
        -0.35 * np.clip(residual_out + residual_in, 0.0, 12.0)
    )
    residual_support = np.clip(residual_support, 0.72, 1.08)

    rank_disagreement = np.abs(finite_rank_out - finite_rank_in)
    bridge_signal = np.tanh(rank_disagreement / rank_scale)
    bridge_signal *= np.exp(
        -0.35 * np.maximum(relative_distance - 1.0, 0.0)
    )
    bridge_term = 1.0 + 0.08 * bridge_signal

    long_edge_penalty = np.exp(
        -0.24 * np.maximum(relative_distance - 1.0, 0.0)
    )

    base = (
        distance_quality
        * (0.60 + 0.24 * np.sqrt(outgoing_quality * incoming_quality)
           + 0.16 * rank_quality)
        * regret_term
        * (1.0 + 0.16 * mutual_nearest)
        * bridge_term
        * residual_support
        * long_edge_penalty
    )

    base[~valid] = 0.0
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

    row_totals = base.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        base,
        row_totals,
        out=np.zeros_like(base),
        where=row_totals > 0.0,
    )

    entropy = np.zeros(n, dtype=float)
    positive = probabilities > 0.0
    entropy[positive.any(axis=1)] = -np.sum(
        np.where(positive, probabilities * np.log(np.maximum(probabilities, eps)), 0.0),
        axis=1,
    )[positive.any(axis=1)]

    entropy_scale = np.log(max(2, n - 1))
    concentration = 1.0 - np.clip(
        entropy / max(entropy_scale, eps),
        0.0,
        1.0,
    )
    exploration_strength = 0.025 + 0.075 * float(np.mean(concentration))

    diversity_term = np.exp(
        -0.35 * (
            finite_rank_out + finite_rank_in
        ) / rank_scale
    )
    diversity_term = np.where(valid, diversity_term, 0.0)

    mixed = (
        (1.0 - exploration_strength) * base
        + exploration_strength * (
            np.maximum(base.max(axis=1, keepdims=True), eps)
            * (0.35 + 0.65 * diversity_term)
        )
    )

    mixed[~valid] = 0.0
    np.fill_diagonal(mixed, 0.0)
    mixed = np.nan_to_num(mixed, nan=0.0, posinf=0.0, neginf=0.0)

    totals = mixed.sum(axis=1, keepdims=True)
    return np.divide(
        mixed,
        totals,
        out=np.zeros_like(mixed),
        where=totals > 0.0,
    )
