import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse and long-horizon edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & np.isfinite(raw.T)
        & (raw.T >= 0.0)
        & ~diagonal
    )

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distances, np.inf)

    finite_values = distances[valid]
    positive = finite_values[finite_values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    clean = finite_values
    upper = float(np.quantile(clean, 0.90))
    if not np.isfinite(upper):
        upper = scale

    penalty = max(upper, scale) + 12.0 * scale
    safe = np.where(valid, distances, penalty)
    np.fill_diagonal(safe, penalty)

    valid_counts = valid.sum(axis=1).astype(float)
    positive_counts = valid_counts[valid_counts > 0.0]
    typical_degree = (
        float(np.median(positive_counts))
        if positive_counts.size
        else 1.0
    )

    # Adaptive local temperature: sparse nodes receive a softer distribution.
    local_scale = np.zeros(n, dtype=float)
    for i in range(n):
        row = distances[i, valid[i]]
        positive_row = row[row > 0.0]
        if positive_row.size:
            local_scale[i] = float(np.median(positive_row))
        elif row.size:
            local_scale[i] = max(float(np.mean(row)), scale)
        else:
            local_scale[i] = scale

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        scale,
    )

    temperature = np.sqrt(local_scale[:, None] * local_scale[None, :])
    temperature = np.maximum(0.62 * temperature, eps)

    affinity = np.zeros((n, n), dtype=float)
    exponent = np.minimum(safe / temperature, 50.0)
    affinity[valid] = np.exp(-exponent[valid])

    # Soft rank signal supplies a stable fallback when distances are nearly tied.
    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.75 * np.sqrt(float(max(n, 2))))
    rank_signal = np.exp(-ranks / rank_scale)
    rank_signal[~valid] = 0.0

    # Estimate each node's future choice reservoir without edge-specific degree correction.
    row_mass = affinity.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)

    row_entropy = np.zeros(n, dtype=float)
    for i in range(n):
        weights = affinity[i, valid[i]]
        if weights.size:
            total = float(np.sum(weights))
            if total > 0.0:
                probabilities = weights / total
                row_entropy[i] = -float(
                    np.sum(probabilities * np.log(probabilities + eps))
                )

    effective_options = np.exp(np.clip(row_entropy, 0.0, 50.0))
    reservoir = np.log1p(effective_options) / np.log1p(
        max(typical_degree, 1.0)
    )
    reservoir = np.clip(reservoir, 0.0, 1.5)

    endpoint_reserve = np.sqrt(
        np.maximum(reservoir[:, None] * reservoir[None, :], 0.0)
    )
    endpoint_reserve[~valid] = 0.0

    # Opportunity cost: an edge is slightly downweighted when it dominates
    # one endpoint's entire local reservoir.
    dominance_i = affinity / row_mass[:, None]
    dominance_j = affinity / row_mass[None, :]
    dominance = np.maximum(dominance_i, dominance_j)
    competition = 1.0 - 0.22 * np.clip(dominance, 0.0, 1.0)
    competition[~valid] = 0.0

    affinity_max = max(float(affinity.max()), eps)
    affinity_norm = affinity / affinity_max

    rank_pair = np.sqrt(
        np.maximum(rank_signal * rank_signal.T, 0.0)
    )

    # Local score: distance quality dominates, while reservoir and competition
    # preserve alternatives useful for later tour completion.
    score = (
        0.58 * affinity_norm
        + 0.16 * rank_pair
        + 0.18 * np.minimum(endpoint_reserve, 1.0)
        + 0.08 * competition
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(score.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)
    score /= maximum

    # Light symmetric balancing promotes global coverage while preserving
    # the local distance structure. This is intentionally short and damped.
    balanced = score.copy()
    for _ in range(8):
        row_sum = balanced.sum(axis=1)
        positive_rows = row_sum[row_sum > eps]

        if positive_rows.size == 0:
            break

        target = float(np.median(positive_rows))
        correction = np.sqrt(
            target / np.maximum(row_sum, eps)
        )
        correction = np.clip(correction, 0.72, 1.38)

        balanced *= correction[:, None] * correction[None, :]
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

        peak = float(balanced.max())
        if np.isfinite(peak) and peak > 0.0:
            balanced /= peak

    result = 0.74 * score + 0.26 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
