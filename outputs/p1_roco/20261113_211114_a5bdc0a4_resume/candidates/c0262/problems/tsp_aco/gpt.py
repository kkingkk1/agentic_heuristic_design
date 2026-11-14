import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed soft-assignment priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(raw) & (raw >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = raw[valid & (raw > 0.0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    observed = raw[valid]

    q10, q50, q90 = np.quantile(
        observed,
        [0.10, 0.50, 0.90],
    )
    spread = max(float(q90 - q10), 0.25 * scale, eps)

    replacement = max(
        float(np.max(observed)),
        float(q90),
        scale,
    ) + 10.0 * spread

    costs = np.where(valid, raw, replacement).astype(float, copy=True)
    np.fill_diagonal(costs, replacement)

    normalized = np.clip(
        (costs - q10) / spread,
        -12.0,
        24.0,
    )

    temperature = max(
        0.72 * spread,
        0.18 * scale,
        eps,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.clip(
            (costs[valid] - q10) / temperature,
            -40.0,
            60.0,
        )
    )

    row_max = np.maximum(
        np.max(affinity, axis=1, keepdims=True),
        eps,
    )
    col_max = np.maximum(
        np.max(affinity, axis=0, keepdims=True),
        eps,
    )

    row_relative = affinity / row_max
    col_relative = affinity / col_max

    row_rank_order = np.argsort(
        costs,
        axis=1,
        kind="stable",
    )
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_rank_order,
    ] = np.arange(n, dtype=float)

    col_rank_order = np.argsort(
        costs,
        axis=0,
        kind="stable",
    )
    col_rank = np.empty((n, n), dtype=float)
    col_rank[
        col_rank_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)

    rank_temperature = max(
        0.80 * np.sqrt(float(n)),
        1.0,
    )

    rank_signal = np.exp(
        -(row_rank + col_rank)
        / (2.0 * rank_temperature)
    )
    rank_signal[~valid] = 0.0

    masked_affinity = affinity.copy()
    masked_affinity[~valid] = 0.0

    assignment = masked_affinity.copy()

    for _ in range(36):
        row_sum = assignment.sum(axis=1, keepdims=True)
        assignment /= np.maximum(row_sum, eps)

        col_sum = assignment.sum(axis=0, keepdims=True)
        assignment /= np.maximum(col_sum, eps)

        assignment[~valid] = 0.0

    row_sum = assignment.sum(axis=1, keepdims=True)
    assignment /= np.maximum(row_sum, eps)

    col_sum = assignment.sum(axis=0, keepdims=True)
    assignment /= np.maximum(col_sum, eps)

    continuation = np.zeros((n, n), dtype=float)

    for i in range(n):
        outgoing = np.flatnonzero(valid[i])
        if outgoing.size == 0:
            continue

        for j in outgoing:
            candidates = np.flatnonzero(valid[j])
            candidates = candidates[candidates != i]

            if candidates.size == 0:
                candidates = np.flatnonzero(valid[j])

            if candidates.size == 0:
                continue

            next_cost = costs[j, candidates]
            next_mass = assignment[j, candidates]

            finite_next = np.isfinite(next_cost) & np.isfinite(next_mass)
            if not np.any(finite_next):
                continue

            next_cost = next_cost[finite_next]
            next_mass = next_mass[finite_next]

            local_scale = max(
                float(np.median(next_cost)),
                0.20 * scale,
                eps,
            )

            continuation[i, j] = float(
                np.sum(
                    next_mass
                    * np.exp(
                        -np.clip(
                            next_cost / local_scale,
                            0.0,
                            40.0,
                        )
                    )
                )
            )

    continuation_max = float(np.max(continuation))
    if np.isfinite(continuation_max) and continuation_max > 0.0:
        continuation /= continuation_max

    row_competition = np.zeros((n, n), dtype=float)

    for i in range(n):
        values = costs[i, valid[i]]
        if values.size == 0:
            continue

        best = float(np.min(values))
        local_scale = max(
            float(np.median(np.abs(values - best))),
            0.15 * spread,
            eps,
        )

        row_competition[i] = np.exp(
            -np.clip(
                (costs[i] - best) / local_scale,
                0.0,
                40.0,
            )
        )

    row_competition[~valid] = 0.0

    global_signal = np.zeros((n, n), dtype=float)
    finite_values = np.sort(observed)

    ranks = np.searchsorted(
        finite_values,
        raw,
        side="right",
    ).astype(float)

    global_signal[valid] = 1.0 - (
        ranks[valid] / max(float(finite_values.size), 1.0)
    )
    global_signal[valid] = 0.25 + 0.75 * global_signal[valid]

    assignment_max = float(np.max(assignment))
    if np.isfinite(assignment_max) and assignment_max > 0.0:
        assignment_signal = assignment / assignment_max
    else:
        assignment_signal = np.zeros_like(assignment)

    score = (
        0.40 * assignment_signal
        + 0.22 * continuation
        + 0.16 * np.sqrt(
            np.maximum(row_relative * col_relative, 0.0)
        )
        + 0.12 * row_competition
        + 0.10 * rank_signal * global_signal
    )

    score[~valid] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score = valid.astype(float)
        np.fill_diagonal(score, 0.0)

        fallback_max = float(np.max(score))
        if fallback_max > 0.0:
            score /= fallback_max

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
