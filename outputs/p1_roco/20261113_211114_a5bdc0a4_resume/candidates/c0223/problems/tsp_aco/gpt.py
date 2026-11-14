import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & ~diagonal
    valid = finite | finite.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    nonnegative = np.where(np.isfinite(raw), np.maximum(raw, 0.0), 0.0)
    available = finite.astype(float)

    distance = nonnegative + nonnegative.T
    counts = available + available.T
    distance = np.divide(
        distance,
        counts,
        out=np.zeros_like(distance),
        where=counts > 0.0,
    )
    distance = 0.5 * (distance + distance.T)

    observed = distance[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    largest = float(np.max(observed)) if observed.size else scale
    penalty = max(largest + 10.0 * scale, scale)

    distance = np.where(valid, distance, penalty)
    np.fill_diagonal(distance, penalty)

    valid_values = distance[valid]
    temperature = (
        float(np.median(valid_values)) if valid_values.size else scale
    )
    temperature = max(0.30 * temperature, np.finfo(float).eps)

    row_best = np.min(
        np.where(valid, distance, penalty),
        axis=1,
        keepdims=True,
    )

    centered = np.maximum(distance - row_best, 0.0)
    affinity = np.exp(-centered / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_affinity = affinity.sum(axis=1, keepdims=True)
    row_affinity = np.maximum(row_affinity, np.finfo(float).eps)
    normalized_affinity = affinity / row_affinity

    endpoint_preference = np.sqrt(
        np.maximum(normalized_affinity, 0.0)
        * np.maximum(normalized_affinity.T, 0.0)
    )
    endpoint_preference[~valid] = 0.0

    sorted_distance = np.sort(
        np.where(valid, distance, penalty),
        axis=1,
    )
    nearest = sorted_distance[:, 0]
    second_nearest = sorted_distance[:, 1]

    regret = np.maximum(second_nearest - nearest, 0.0)
    regret_scale = np.maximum(0.18 * scale, 0.35 * regret)
    regret_scale = np.maximum(regret_scale, np.finfo(float).eps)

    local_competition = np.exp(
        -np.maximum(distance - nearest[:, None], 0.0)
        / regret_scale[:, None]
    )
    local_competition[~valid] = 0.0
    local_competition = np.sqrt(
        np.maximum(local_competition, 0.0)
        * np.maximum(local_competition.T, 0.0)
    )

    bypass = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        candidate = distance[:, k, None] + distance[None, k, :]
        bypass = np.minimum(bypass, candidate)

    np.fill_diagonal(bypass, penalty)

    detour_gain = np.maximum(bypass - distance, 0.0)
    detour_scale = max(0.70 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(-detour_gain / detour_scale)
    irreplaceability[~valid] = 0.0

    score = (
        0.46 * endpoint_preference
        + 0.29 * local_competition
        + 0.25 * irreplaceability
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(32):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.62 * score + 0.38 * balanced
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
