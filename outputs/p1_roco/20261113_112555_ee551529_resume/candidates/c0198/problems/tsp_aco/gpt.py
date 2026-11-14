import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return bounded symmetric edge priors for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    both_valid = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = finite.T & ~finite

    cost = np.full((n, n), np.inf, dtype=float)
    cost[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])
    cost[forward_only] = d[forward_only]
    cost[backward_only] = d.T[backward_only]
    np.fill_diagonal(cost, np.inf)

    valid = np.isfinite(cost)
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    positive = cost[valid & (cost > 0.0)]

    if positive.size:
        q25, q50, q75 = np.percentile(positive, [25.0, 50.0, 75.0])
    else:
        q25 = q50 = q75 = 1.0

    global_scale = max(float(q50), eps)
    global_spread = max(float(q75 - q25), 0.25 * global_scale, eps)

    neighborhood = min(max(3, int(np.ceil(np.sqrt(max(n, 2))))), 8, n - 1)

    local_scale = np.full(n, global_scale, dtype=float)
    rank_weight = np.zeros((n, n), dtype=float)
    reserve = np.zeros((n, n), dtype=float)

    for i in range(n):
        row = cost[i]
        finite_indices = np.flatnonzero(np.isfinite(row))

        if finite_indices.size == 0:
            continue

        ordered = finite_indices[
            np.argsort(row[finite_indices], kind="stable")
        ]

        local_indices = ordered[:min(neighborhood, ordered.size)]
        local_values = row[local_indices]

        positive_local = local_values[local_values > 0.0]
        if positive_local.size:
            local_scale[i] = max(
                float(np.mean(positive_local)),
                0.35 * global_scale,
                eps,
            )
        else:
            local_scale[i] = global_scale

        denominator = max(float(neighborhood - 1), 1.0)
        for rank, j in enumerate(ordered[:neighborhood]):
            rank_weight[i, j] = np.exp(-float(rank) / denominator)

        for j in ordered[:min(2, ordered.size)]:
            reserve[i, j] = 1.0

    local_scale = np.maximum(local_scale, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pair_scale = np.sqrt(
            np.maximum(np.outer(local_scale, local_scale), eps)
        )

        local_ratio = cost / pair_scale
        local_shortness = np.exp(-np.clip(local_ratio, 0.0, 40.0))

        global_ratio = (cost - q25) / global_spread
        global_competitiveness = np.exp(
            -np.clip(global_ratio, -4.0, 20.0)
        )

        rank_consensus = np.sqrt(
            np.maximum(rank_weight * rank_weight.T, 0.0)
        )

        reciprocal = (
            (rank_weight > 0.0)
            & (rank_weight.T > 0.0)
        ).astype(float)

        reserve_both = reserve * reserve.T
        reserve_one = np.clip(reserve + reserve.T, 0.0, 1.0)

        # Mild coverage correction: only adjust otherwise competitive edges.
        row_min = np.min(np.where(valid, cost, np.inf), axis=1)
        shifted = np.where(valid, cost - row_min[:, None], np.inf)

        temperature = np.maximum(
            0.75 * local_scale + 0.25 * global_spread,
            eps,
        )

        candidate_mass = np.exp(
            -np.clip(shifted / temperature[:, None], 0.0, 20.0)
        )
        candidate_mass[~valid] = 0.0

        pressure = np.sum(candidate_mass, axis=1)
        active_pressure = pressure[pressure > 0.0]

        mean_pressure = (
            max(float(np.mean(active_pressure)), eps)
            if active_pressure.size
            else 1.0
        )

        pressure_ratio = np.sqrt(
            np.maximum(
                np.outer(pressure, pressure) / (mean_pressure * mean_pressure),
                eps,
            )
        )

        coverage = np.clip(
            1.0 / np.sqrt(pressure_ratio),
            0.85,
            1.15,
        )

        quality_gate = 0.35 + 0.65 * np.sqrt(
            np.maximum(
                local_shortness * global_competitiveness,
                0.0,
            )
        )

        coverage_factor = 1.0 + 0.08 * quality_gate * (coverage - 1.0)

        base = (
            0.70 * local_shortness
            + 0.30 * global_competitiveness
        )

        base *= 0.76 + 0.24 * rank_consensus

        scores = (
            base
            * (1.0 + 0.10 * reciprocal)
            * (1.0 + 0.08 * reserve_both + 0.025 * reserve_one)
            * coverage_factor
        )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
