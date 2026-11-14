import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    only_forward = finite & ~finite.T
    only_backward = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[only_forward] = raw[only_forward]
    cleaned[only_backward] = raw.T[only_backward]
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_values = observed[np.isfinite(observed)]
    largest = float(np.max(finite_values)) if finite_values.size else scale
    penalty = largest + 16.0 * scale

    distances = np.where(valid, cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    valid_values = distances[valid]
    temperature = float(np.median(valid_values)) if valid_values.size else scale
    temperature = max(0.24 * temperature, np.finfo(float).eps)

    affinity = np.exp(-np.minimum(distances, penalty) / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    max_affinity = float(np.max(affinity))
    if not np.isfinite(max_affinity) or max_affinity <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    base = affinity / max_affinity

    row_sorted = np.sort(
        np.where(valid, distances, penalty),
        axis=1,
    )
    nearest = row_sorted[:, 0]
    second = row_sorted[:, 1] if n > 2 else nearest
    local_gap = np.maximum(second - nearest, 0.14 * scale)
    local_gap = np.maximum(local_gap, np.finfo(float).eps)

    regret = np.exp(
        np.clip((second[:, None] - distances) / local_gap[:, None], -18.0, 18.0)
    )
    regret[~valid] = 0.0
    np.fill_diagonal(regret, 0.0)

    local_preferences = np.exp(
        np.clip(-(distances - nearest[:, None]) / local_gap[:, None], -24.0, 0.0)
    )
    local_preferences[~valid] = 0.0
    np.fill_diagonal(local_preferences, 0.0)

    preference_max = np.maximum(local_preferences.max(axis=1, keepdims=True), 1e-15)
    local_preferences /= preference_max

    endpoint_competition = np.sqrt(
        np.maximum(local_preferences, 0.0)
        * np.maximum(local_preferences.T, 0.0)
    )
    endpoint_regret = np.sqrt(
        np.maximum(regret, 0.0) * np.maximum(regret.T, 0.0)
    )

    best_two_hop = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        via_k = distances[:, k, None] + distances[None, k, :]
        best_two_hop = np.minimum(best_two_hop, via_k)

    np.fill_diagonal(best_two_hop, penalty)
    bypass_gain = np.maximum(best_two_hop - distances, 0.0)
    bypass_scale = max(0.68 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(-bypass_gain / bypass_scale)
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    score = (
        0.42 * base
        + 0.24 * endpoint_competition
        + 0.18 * endpoint_regret
        + 0.16 * irreplaceability
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(32):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

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
