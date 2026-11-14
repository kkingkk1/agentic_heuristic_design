import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = offdiag & (finite | finite.T)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both_finite = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    cleaned[both_finite] = 0.5 * (raw[both_finite] + raw.T[both_finite])
    cleaned[forward_only] = raw[forward_only]
    cleaned[backward_only] = raw.T[backward_only]
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    penalty = maximum + 16.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = cleaned[valid]
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps * max(scale, 1.0)

    temperature = max(0.55 * scale, eps)
    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / temperature, 60.0)
    )

    affinity_max = float(np.max(affinity))
    if np.isfinite(affinity_max) and affinity_max > 0.0:
        affinity /= affinity_max

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (distances[valid] + eps)

    row_max = np.maximum(inverse.max(axis=1, keepdims=True), eps)
    relative_closeness = inverse / row_max

    row_mass = np.maximum(affinity.sum(axis=1, keepdims=True), eps)
    local_probability = affinity / row_mass

    mutual_preference = np.sqrt(
        np.maximum(local_probability * local_probability.T, 0.0)
    )

    nearest = distances.min(axis=1)
    second = np.partition(distances, 1, axis=1)[:, 1]
    local_scale = np.maximum(second - nearest, 0.18 * scale)
    local_scale = np.maximum(local_scale, eps)

    nearest_mask = distances <= nearest[:, None] + eps
    local_gap = np.where(
        nearest_mask,
        second[:, None] - distances,
        nearest[:, None] - distances,
    )
    endpoint_competition = np.exp(
        np.clip(local_gap / local_scale[:, None], -24.0, 24.0)
    )
    endpoint_competition[~valid] = 0.0
    endpoint_competition = np.sqrt(
        np.maximum(endpoint_competition * endpoint_competition.T, 0.0)
    )

    weighted_degree = np.maximum(affinity.sum(axis=1), eps)
    scarcity = 1.0 / np.sqrt(weighted_degree)
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.45, 2.25)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )

    bypass = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        candidate = distances[:, k, None] + distances[None, k, :]
        bypass = np.minimum(bypass, candidate)

    np.fill_diagonal(bypass, penalty)
    bypass_gain = np.maximum(bypass - distances, 0.0)
    bypass_scale = max(0.78 * scale, eps)
    irreplaceability = 1.0 - np.exp(-bypass_gain / bypass_scale)
    irreplaceability[~valid] = 0.0

    score = (
        0.30 * affinity
        + 0.18 * relative_closeness
        + 0.20 * mutual_preference
        + 0.16 * endpoint_competition
        + 0.10 * irreplaceability
        + 0.06 * np.minimum(endpoint_scarcity, 2.0) / 2.0
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()
    for _ in range(36):
        row_sum = np.maximum(balanced.sum(axis=1, keepdims=True), eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max
    else:
        balanced.fill(0.0)

    result = 0.64 * score + 0.36 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum_result = float(np.max(result))
    if np.isfinite(maximum_result) and maximum_result > 0.0:
        result /= maximum_result
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
