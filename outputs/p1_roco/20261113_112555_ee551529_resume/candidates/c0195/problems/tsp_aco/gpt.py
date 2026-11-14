import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse and long-term-aware TSP edge priors."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    reverse_finite = finite.T
    both_valid = finite & reverse_finite
    forward_only = finite & ~reverse_finite
    backward_only = reverse_finite & ~finite

    cost = np.full((n, n), np.inf, dtype=float)
    cost[both_valid] = 0.5 * (d[both_valid] + d.T[both_valid])
    cost[forward_only] = d[forward_only]
    cost[backward_only] = d.T[backward_only]
    np.fill_diagonal(cost, np.inf)

    valid = np.isfinite(cost)
    if not np.any(valid):
        return result

    positive = cost[valid & (cost > 0.0)]
    eps = np.finfo(float).eps

    if positive.size:
        q10, q25, q50, q75, q90 = np.percentile(
            positive, [10.0, 25.0, 50.0, 75.0, 90.0]
        )
    else:
        q10 = q25 = q50 = q75 = q90 = 1.0

    global_scale = max(float(q50), eps)
    global_spread = max(float(q75 - q25), 0.25 * global_scale, eps)

    k = min(max(3, int(np.ceil(np.sqrt(max(n, 2))))), 8, n - 1)
    local_scale = np.full(n, global_scale, dtype=float)
    first = np.full(n, global_scale, dtype=float)
    second = np.full(n, global_scale, dtype=float)
    rank_weight = np.zeros((n, n), dtype=float)
    reserve_mask = np.zeros((n, n), dtype=float)

    for i in range(n):
        row = cost[i]
        values = np.sort(row[np.isfinite(row)])

        if values.size == 0:
            continue

        first[i] = max(float(values[0]), eps)
        second[i] = max(
            float(values[min(1, values.size - 1)]),
            eps,
        )

        neighborhood = values[:k]
        local_scale[i] = max(
            float(np.median(neighborhood)),
            0.35 * global_scale,
            eps,
        )

        order = np.argsort(
            np.where(np.isfinite(row), row, np.inf),
            kind="stable",
        )

        for rank, j in enumerate(order[:k]):
            if np.isfinite(row[j]):
                rank_weight[i, j] = np.exp(
                    -float(rank) / max(float(k - 1), 1.0)
                )

        top_count = min(2, values.size)
        for j in order[:top_count]:
            if np.isfinite(row[j]):
                reserve_mask[i, j] = 1.0

    local_scale = np.maximum(local_scale, eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        local_ratio = cost / np.sqrt(
            np.outer(local_scale, local_scale)
        )
        global_ratio = cost / global_scale

        local_shortness = np.exp(
            -np.clip(local_ratio, 0.0, 30.0)
        )

        global_shortness = np.exp(
            -np.clip((cost - q10) / global_spread, -5.0, 30.0)
        )

        rank_consensus = np.sqrt(
            np.maximum(rank_weight * rank_weight.T, 0.0)
        )

        # Soft candidate quality used to estimate how strongly each node
        # dominates the global candidate pool.
        temperature = np.maximum(
            0.65 * local_scale + 0.35 * global_spread,
            eps,
        )
        shifted_cost = cost - np.nanmin(
            np.where(valid, cost, np.nan),
            axis=1,
        )[:, None]
        shifted_cost = np.where(valid, shifted_cost, np.inf)

        candidate_mass = np.exp(
            -np.clip(shifted_cost / temperature[:, None], 0.0, 25.0)
        )
        candidate_mass[~valid] = 0.0

        node_pressure = np.sum(candidate_mass, axis=1)
        mean_pressure = max(
            float(np.mean(node_pressure[node_pressure > 0.0]))
            if np.any(node_pressure > 0.0)
            else 1.0,
            eps,
        )

        pressure_ratio = np.sqrt(
            np.outer(
                np.maximum(node_pressure / mean_pressure, eps),
                np.maximum(node_pressure / mean_pressure, eps),
            )
        )

        # Coverage correction is active mainly for globally competitive
        # edges, preventing weak edges from receiving a diversity bonus.
        quality_gate = 0.35 + 0.65 * np.sqrt(
            np.maximum(local_shortness * global_shortness, 0.0)
        )
        coverage_correction = np.clip(
            1.0 / np.sqrt(np.maximum(pressure_ratio, eps)),
            0.78,
            1.22,
        )
        diversity_factor = 1.0 + (
            0.16 * quality_gate * (coverage_correction - 1.0)
        )

        # Explicit two-neighbor reservation replaces unstable opportunity
        # ratios. Edges useful to both endpoints receive a bounded bonus.
        reserve_both = reserve_mask * reserve_mask.T
        reserve_one = np.clip(
            reserve_mask + reserve_mask.T,
            0.0,
            1.0,
        )
        reserve_factor = (
            1.0
            + 0.12 * reserve_both
            + 0.035 * (reserve_one - reserve_both)
        )

        # Mildly prefer edges whose endpoints have comparable local scales.
        scale_balance = np.minimum(
            local_scale[:, None] / np.maximum(local_scale[None, :], eps),
            local_scale[None, :] / np.maximum(local_scale[:, None], eps),
        )
        scale_balance = np.sqrt(np.clip(scale_balance, 0.0, 1.0))
        balance_factor = 0.94 + 0.06 * scale_balance

        # Suppress globally weak edges without creating a hard threshold.
        quantile_gate = 0.88 + 0.12 * np.exp(
            -np.clip((cost - q50) / global_spread, 0.0, 12.0)
        )

        # Favor edges that are locally good and globally competitive, while
        # retaining a small rank-based contribution for structural diversity.
        base = (
            0.58 * local_shortness
            + 0.42 * global_shortness
        )
        base *= 0.68 + 0.32 * rank_consensus

        scores = (
            base
            * diversity_factor
            * reserve_factor
            * balance_factor
            * quantile_gate
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
