import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diversified edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]

    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    valid = finite.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    clean = np.where(finite, np.maximum(distances, 0.0), np.inf)
    np.fill_diagonal(clean, np.inf)

    reciprocal = valid & valid.T
    symmetric = bool(
        np.all(reciprocal == valid)
        and np.allclose(
            clean[reciprocal],
            clean.T[reciprocal],
            rtol=1e-8,
            atol=1e-12,
        )
    )

    if symmetric:
        clean = np.where(
            reciprocal,
            0.5 * (clean + clean.T),
            clean,
        )
        valid = reciprocal.copy()
        np.fill_diagonal(valid, False)

    finite_values = clean[valid]

    positive = finite_values[finite_values > 0.0]
    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        scale = max(median, 1.4826 * mad, 1.0)
    elif finite_values.size:
        scale = max(float(np.median(finite_values)), 1.0)
    else:
        scale = 1.0

    eps = np.sqrt(np.finfo(float).eps) * scale

    normalized = np.zeros((n, n), dtype=float)
    normalized[valid] = np.clip(
        clean[valid] / (scale + eps),
        0.0,
        50.0,
    )

    finite_count = int(finite_values.size)
    density = finite_count / max(n * (n - 1), 1)

    # Mildly adaptive temperature: sparse graphs retain more alternatives.
    temperature = 0.72 + 0.35 * (1.0 - density)
    attraction = np.zeros((n, n), dtype=float)
    attraction[valid] = (
        0.62 * np.exp(-normalized[valid] / temperature)
        + 0.38 / (1.0 + normalized[valid])
    )

    # Relative regret rewards an edge only when it is better than its alternatives.
    row_sorted = np.sort(
        np.where(valid, clean, np.inf),
        axis=1,
    )
    col_sorted = np.sort(
        np.where(valid, clean, np.inf),
        axis=0,
    )

    row_best = row_sorted[:, 0]
    col_best = col_sorted[0, :]

    row_second = row_sorted[:, 1] if n > 1 else row_best
    col_second = col_sorted[1, :] if n > 1 else col_best

    row_regret = np.zeros((n, n), dtype=float)
    col_regret = np.zeros((n, n), dtype=float)

    row_gap = np.zeros(n, dtype=float)
    col_gap = np.zeros(n, dtype=float)

    row_gap[np.isfinite(row_second)] = np.maximum(
        row_second[np.isfinite(row_second)]
        - row_best[np.isfinite(row_second)],
        0.0,
    )
    col_gap[np.isfinite(col_second)] = np.maximum(
        col_second[np.isfinite(col_second)]
        - col_best[np.isfinite(col_second)],
        0.0,
    )

    row_regret[valid] = np.maximum(
        row_gap[:, None] / (scale + eps)
        - normalized[valid],
        0.0,
    )
    col_regret[valid] = np.maximum(
        col_gap[None, :] / (scale + eps)
        - normalized[valid],
        0.0,
    )

    row_regret = np.clip(row_regret, 0.0, 8.0)
    col_regret = np.clip(col_regret, 0.0, 8.0)

    scarcity_bonus = (
        1.0
        + 0.20 * np.tanh(row_regret)
        + 0.20 * np.tanh(col_regret)
    )

    # Restricted two-hop bridge gain.
    candidate_count = min(
        max(2, int(np.sqrt(max(n, 2))) + 1),
        max(n - 1, 1),
    )

    two_hop = np.full((n, n), np.inf, dtype=float)

    for i in range(n):
        outgoing = np.flatnonzero(valid[i])
        if outgoing.size == 0:
            continue

        order = outgoing[np.argsort(clean[i, outgoing])]
        intermediates = order[:candidate_count]

        for middle in intermediates:
            via = clean[i, middle] + clean[middle]
            two_hop[i] = np.minimum(two_hop[i], via)

    np.fill_diagonal(two_hop, np.inf)

    bridge_gain = np.zeros((n, n), dtype=float)
    bridge_valid = valid & np.isfinite(two_hop)

    bridge_gain[bridge_valid] = np.maximum(
        two_hop[bridge_valid] - clean[bridge_valid],
        0.0,
    ) / (scale + eps)
    bridge_gain = np.clip(bridge_gain, 0.0, 8.0)

    bridge_weight = 0.12 + 0.08 * density
    bridge_bonus = 1.0 + bridge_weight * np.tanh(bridge_gain)

    raw = attraction * scarcity_bonus * bridge_bonus
    raw[~valid] = 0.0

    # Estimate future option capacity at each endpoint.
    outgoing_mass = raw.sum(axis=1)
    incoming_mass = raw.sum(axis=0)

    def normalized_capacity(values: np.ndarray) -> np.ndarray:
        active = values > 0.0
        result = np.zeros_like(values)
        if not np.any(active):
            return result

        reference = float(np.median(values[active]))
        result[active] = np.sqrt(
            np.clip(
                values[active] / (reference + eps),
                0.25,
                4.0,
            )
        )
        return result

    outgoing_capacity = normalized_capacity(outgoing_mass)
    incoming_capacity = normalized_capacity(incoming_mass)

    # Entropy-like support breadth rewards endpoints with diverse alternatives.
    support = np.zeros((n, n), dtype=float)
    support[valid] = attraction[valid]

    row_support = support.sum(axis=1)
    col_support = support.sum(axis=0)

    row_entropy = np.zeros(n, dtype=float)
    col_entropy = np.zeros(n, dtype=float)

    for i in range(n):
        values = support[i]
        total = float(values.sum())
        if total > 0.0:
            probabilities = values[values > 0.0] / total
            row_entropy[i] = -float(
                np.sum(probabilities * np.log(probabilities + eps))
            )

    for j in range(n):
        values = support[:, j]
        total = float(values.sum())
        if total > 0.0:
            probabilities = values[values > 0.0] / total
            col_entropy[j] = -float(
                np.sum(probabilities * np.log(probabilities + eps))
            )

    active_row_entropy = row_entropy[row_support > 0.0]
    active_col_entropy = col_entropy[col_support > 0.0]

    row_entropy_scale = (
        float(np.median(active_row_entropy))
        if active_row_entropy.size
        else 1.0
    )
    col_entropy_scale = (
        float(np.median(active_col_entropy))
        if active_col_entropy.size
        else 1.0
    )

    row_breadth = np.sqrt(
        np.clip(
            row_entropy / (row_entropy_scale + eps),
            0.35,
            2.5,
        )
    )
    col_breadth = np.sqrt(
        np.clip(
            col_entropy / (col_entropy_scale + eps),
            0.35,
            2.5,
        )
    )

    future_potential = (
        incoming_capacity[:, None]
        * outgoing_capacity[None, :]
        * row_breadth[:, None]
        * col_breadth[None, :]
    )

    active_potential = future_potential[valid]
    if active_potential.size:
        potential_scale = float(np.median(active_potential))
        potential_factor = np.sqrt(
            np.clip(
                future_potential / (potential_scale + eps),
                0.45,
                2.2,
            )
        )
    else:
        potential_factor = np.ones((n, n), dtype=float)

    raw *= 1.0 + 0.16 * np.tanh(potential_factor - 1.0)
    raw[~valid] = 0.0

    if symmetric:
        raw = 0.5 * (raw + raw.T)
        raw[~valid] = 0.0

    # Damped global balancing prevents a few endpoints from monopolizing mass.
    corrected = raw.copy()

    for _ in range(5):
        row_sum = corrected.sum(axis=1)
        active_rows = row_sum > 0.0

        if np.any(active_rows):
            target_row = float(np.median(row_sum[active_rows]))
            factor = np.ones(n, dtype=float)
            factor[active_rows] = np.sqrt(
                np.clip(
                    target_row / (row_sum[active_rows] + eps),
                    0.65,
                    1.55,
                )
            )
            corrected *= factor[:, None]

        col_sum = corrected.sum(axis=0)
        active_cols = col_sum > 0.0

        if np.any(active_cols):
            target_col = float(np.median(col_sum[active_cols]))
            factor = np.ones(n, dtype=float)
            factor[active_cols] = np.sqrt(
                np.clip(
                    target_col / (col_sum[active_cols] + eps),
                    0.65,
                    1.55,
                )
            )
            corrected *= factor[None, :]

        corrected[~valid] = 0.0

    if symmetric:
        corrected = 0.5 * (corrected + corrected.T)
        corrected[~valid] = 0.0

    # Adaptive concave compression preserves near-optimal exploratory edges.
    active_values = corrected[valid]
    if active_values.size:
        spread = float(np.std(active_values))
        center = float(np.mean(active_values))
        variability = spread / (center + eps)
        exponent = float(np.clip(0.88 - 0.08 * variability, 0.68, 0.88))
    else:
        exponent = 0.82

    priors = np.power(np.maximum(corrected, 0.0), exponent)
    priors[~valid] = 0.0

    maximum = float(np.max(priors))
    if maximum > 0.0 and np.isfinite(maximum):
        priors /= maximum

    priors[~np.isfinite(priors)] = 0.0
    np.fill_diagonal(priors, 0.0)

    if symmetric:
        priors = 0.5 * (priors + priors.T)
        priors[~valid] = 0.0

    return priors
