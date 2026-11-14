import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-prior matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    distance = np.full((n, n), np.inf, dtype=float)
    distance[valid] = 0.5 * (raw[valid] + raw.T[valid])
    np.fill_diagonal(distance, np.inf)

    values = distance[valid]
    positive_values = values[values > 0.0]

    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    finite_values = values[np.isfinite(values)]
    fallback = (
        float(np.max(finite_values)) + 8.0 * scale
        if finite_values.size
        else 8.0 * scale
    )

    clean = distance.copy()
    clean[~valid] = fallback
    np.fill_diagonal(clean, fallback)

    temperature = max(0.58 * scale, eps)

    attraction = np.zeros((n, n), dtype=float)
    attraction[valid] = np.exp(
        -np.minimum(clean[valid] / temperature, 40.0)
    )

    order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)

    rank_width = max(1.0, 0.78 * np.sqrt(float(max(n - 1, 1))))
    rank_preference = np.exp(-ranks / rank_width)
    rank_preference[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_preference * rank_preference.T, 0.0)
    )

    nearest_count = max(1, min(5, n - 1))
    nearest = np.where(valid, clean, np.inf)
    nearest = np.sort(nearest, axis=1)[:, :nearest_count]
    nearest = np.where(np.isfinite(nearest), nearest, fallback)

    best = nearest[:, 0]
    if nearest_count >= 2:
        second = nearest[:, 1]
    else:
        second = best + scale

    local_gap = np.maximum(second - best, 0.18 * scale)

    advantage = np.maximum(second[:, None] - clean, 0.0)
    endpoint_advantage = 1.0 - np.exp(
        -advantage / (local_gap[:, None] + eps)
    )
    endpoint_advantage[~valid] = 0.0

    mutual_advantage = np.sqrt(
        np.maximum(
            endpoint_advantage * endpoint_advantage.T,
            0.0,
        )
    )

    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, eps)

    sparsity = local_scale / (local_scale + scale)
    sparsity = np.clip(sparsity, 0.0, 0.75)

    sparse_pair = np.sqrt(
        sparsity[:, None] * sparsity[None, :]
    )
    sparse_pair[~valid] = 0.0

    unilateral_protection = (
        0.65 * np.maximum(rank_preference, rank_preference.T)
        + 0.35 * np.maximum(
            endpoint_advantage,
            endpoint_advantage.T,
        )
    )
    unilateral_protection[~valid] = 0.0

    row_mass = attraction.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)
    probabilities = attraction / row_mass[:, None]

    concentration = np.sum(probabilities * probabilities, axis=1)
    flexibility = 1.0 - np.sqrt(
        np.clip(concentration, 0.0, 1.0)
    )
    flexibility = np.clip(flexibility, 0.05, 1.0)

    reserve_pair = np.sqrt(
        flexibility[:, None] * flexibility[None, :]
    )
    reserve_pair[~valid] = 0.0

    score = (
        0.55 * attraction
        + 0.20 * mutual_rank
        + 0.14 * mutual_advantage
        + 0.06 * unilateral_protection
        + 0.05 * sparse_pair
    )

    score *= 0.975 + 0.025 * reserve_pair
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return result

    score /= maximum

    row_sum = score.sum(axis=1)
    positive_rows = row_sum[row_sum > 0.0]
    target = (
        float(np.median(positive_rows))
        if positive_rows.size
        else 1.0
    )
    target = max(target, eps)

    correction = np.sqrt(
        target / np.maximum(row_sum, eps)
    )
    correction = np.clip(correction, 0.88, 1.14)

    corrected = score * np.sqrt(
        correction[:, None] * correction[None, :]
    )
    corrected[~valid] = 0.0
    corrected = 0.5 * (corrected + corrected.T)
    np.fill_diagonal(corrected, 0.0)

    result = 0.84 * score + 0.16 * corrected
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
