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

    finite = np.isfinite(raw)
    off_diag = ~np.eye(n, dtype=bool)
    valid = (finite | finite.T) & off_diag

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T & off_diag
    forward = finite & ~finite.T & off_diag
    backward = ~finite & finite.T & off_diag

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[forward] = raw[forward]
    cleaned[backward] = raw.T[backward]
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    largest = float(np.max(observed)) if observed.size else scale
    penalty = largest + 32.0 * scale

    distances = np.where(valid, cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    valid_values = distances[valid]
    median_distance = float(np.median(valid_values))
    temperature = max(0.30 * max(median_distance, scale), np.finfo(float).eps)

    affinity = np.exp(
        -np.clip((distances - median_distance) / temperature, -24.0, 24.0)
    )
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_max = np.maximum(affinity.max(axis=1, keepdims=True), 1e-15)
    short_edge = affinity / row_max

    ranked = np.where(valid, distances, penalty)
    ranked = np.sort(ranked, axis=1)
    nearest = ranked[:, 0]
    second = ranked[:, 1] if n > 2 else nearest

    gap = np.maximum(second - nearest, 0.12 * scale)
    gap = np.maximum(gap, np.finfo(float).eps)

    local_preference = np.exp(
        -np.clip(
            (distances - nearest[:, None]) / gap[:, None],
            0.0,
            24.0,
        )
    )
    local_preference[~valid] = 0.0
    np.fill_diagonal(local_preference, 0.0)

    preference_max = np.maximum(
        local_preference.max(axis=1, keepdims=True),
        1e-15,
    )
    local_preference /= preference_max

    mutual_preference = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )

    rank_score = np.zeros((n, n), dtype=float)
    for i in range(n):
        valid_indices = np.flatnonzero(valid[i])
        if valid_indices.size:
            order = valid_indices[np.argsort(distances[i, valid_indices])]
            rank_score[i, order] = 1.0 - (
                np.arange(order.size, dtype=float)
                / max(order.size - 1, 1)
            )

    rank_score = np.sqrt(
        np.maximum(rank_score, 0.0) * np.maximum(rank_score.T, 0.0)
    )

    best_two_hop = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        if k == 0:
            via = distances[:, k, None] + distances[None, k, :]
        else:
            via = distances[:, k, None] + distances[None, k, :]
        best_two_hop = np.minimum(best_two_hop, via)

    np.fill_diagonal(best_two_hop, penalty)
    bypass_gain = np.maximum(best_two_hop - distances, 0.0)
    bypass_scale = max(0.75 * scale, np.finfo(float).eps)
    structural_gain = 1.0 - np.exp(-bypass_gain / bypass_scale)
    structural_gain[~valid] = 0.0
    np.fill_diagonal(structural_gain, 0.0)

    score = (
        0.44 * short_edge
        + 0.22 * mutual_preference
        + 0.16 * rank_score
        + 0.18 * structural_gain
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(24):
        row_sum = balanced.sum(axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, np.finfo(float).eps)
        balanced = 0.5 * (balanced + balanced.T)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.64 * score + 0.36 * balanced
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
