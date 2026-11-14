import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric structural edge priors for stochastic TSP sampling."""
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

    edge_values = distances[valid]
    positive_values = edge_values[edge_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    finite_rows = np.any(valid, axis=1)
    row_min = np.min(
        np.where(valid, distances, np.inf),
        axis=1,
    )
    row_min = np.where(
        np.isfinite(row_min),
        row_min,
        scale,
    )

    second = np.partition(
        np.where(valid, distances, np.inf),
        kth=min(1, n - 1),
        axis=1,
    )[:, min(1, n - 1)]
    second = np.where(
        np.isfinite(second),
        second,
        row_min + scale,
    )
    gap = np.maximum(second - row_min, 0.12 * scale)
    gap = np.where(np.isfinite(gap), gap, scale)

    normalized = np.where(valid, distances / scale, np.inf)

    temperature = max(0.62, 4.0 * np.finfo(float).eps)
    shifted = np.where(
        valid,
        -(normalized - row_min[:, None] / scale) / temperature,
        -np.inf,
    )
    shifted = np.clip(shifted, -50.0, 0.0)

    soft_preference = np.zeros((n, n), dtype=float)
    soft_preference[valid] = np.exp(shifted[valid])

    row_mass = soft_preference.sum(axis=1)
    row_mass = np.maximum(row_mass, eps)
    row_probability = soft_preference / row_mass[:, None]

    mutual_probability = np.sqrt(
        np.maximum(row_probability * row_probability.T, 0.0)
    )
    mutual_probability[~valid] = 0.0

    rank_order = np.argsort(distances, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], rank_order] = np.arange(n)

    rank_scale = max(1.0, 0.75 * np.sqrt(float(n)))
    rank_preference = np.exp(-rank / rank_scale)
    rank_preference[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_preference * rank_preference.T, 0.0)
    )

    endpoint_gap = np.maximum(
        second[:, None] - distances,
        0.0,
    )
    endpoint_gap = np.where(
        valid,
        endpoint_gap / gap[:, None],
        0.0,
    )
    endpoint_pressure = 1.0 - np.exp(
        -np.clip(endpoint_gap, 0.0, 24.0)
    )
    endpoint_pressure[~valid] = 0.0

    scarcity = 1.0 / np.sqrt(np.maximum(row_mass, eps))
    scarcity /= max(float(np.mean(scarcity[finite_rows])), eps)
    scarcity = np.clip(scarcity, 0.70, 1.70)

    pair_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    pair_scarcity[~valid] = 0.0

    positive_distance = np.zeros((n, n), dtype=float)
    positive_distance[valid] = 1.0 / (
        1.0 + np.maximum(distances[valid], 0.0) / scale
    )

    local_signal = (
        0.52 * mutual_probability
        + 0.30 * mutual_rank
        + 0.18 * pair_scarcity
    )

    structural_signal = (
        0.72 * endpoint_pressure
        + 0.28 * pair_scarcity
    )

    score = (
        np.maximum(positive_distance, eps) ** 0.92
        * np.maximum(0.25 + local_signal, eps) ** 0.86
        * np.maximum(0.55 + structural_signal, eps) ** 0.58
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.zeros((n, n), dtype=float)

    score /= maximum

    balanced = score.copy()
    for _ in range(16):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    result = 0.68 * score + 0.32 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
