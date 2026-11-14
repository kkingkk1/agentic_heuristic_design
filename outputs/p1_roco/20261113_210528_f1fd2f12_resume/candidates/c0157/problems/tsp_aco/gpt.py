import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    nonnegative = distances >= 0.0
    directed_valid = finite & nonnegative
    np.fill_diagonal(directed_valid, False)

    valid = directed_valid & directed_valid.T
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)
    symmetric[valid] = 0.5 * (
        distances[valid] + distances.T[valid]
    )
    np.fill_diagonal(symmetric, np.inf)

    edge_values = symmetric[valid]
    positive = edge_values[edge_values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = max(scale * 1e-12, np.finfo(float).eps)
    finite_max = float(np.max(edge_values))
    replacement = finite_max + 16.0 * scale

    clean = symmetric.copy()
    clean[~valid] = replacement
    np.fill_diagonal(clean, replacement)

    inverse = 1.0 / (clean + eps)
    inverse[~valid] = 0.0
    np.fill_diagonal(inverse, 0.0)

    row_peak = np.maximum(np.max(inverse, axis=1, keepdims=True), eps)
    col_peak = np.maximum(np.max(inverse, axis=0, keepdims=True), eps)

    row_competition = inverse / row_peak
    col_competition = inverse / col_peak
    competition = np.sqrt(
        np.maximum(row_competition * col_competition, 0.0)
    )

    proximity = np.exp(
        -np.minimum(clean / max(0.90 * scale, eps), 60.0)
    )
    proximity[~valid] = 0.0
    np.fill_diagonal(proximity, 0.0)

    row_mass = np.maximum(proximity.sum(axis=1), eps)
    local_preference = proximity / row_mass[:, None]
    mutual_support = np.sqrt(
        np.maximum(local_preference * local_preference.T, 0.0)
    )

    scarcity = 1.0 / row_mass
    scarcity_mean = float(np.mean(scarcity))
    if np.isfinite(scarcity_mean) and scarcity_mean > eps:
        scarcity /= scarcity_mean
    scarcity = np.clip(scarcity, 0.25, 4.0)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    endpoint_scarcity = np.clip(endpoint_scarcity, 0.5, 2.5)

    best_two_hop = np.full((n, n), replacement, dtype=float)

    for k in range(n):
        candidate = clean[:, k, None] + clean[None, k, :]
        candidate[~valid] = replacement
        best_two_hop = np.minimum(best_two_hop, candidate)

    best_two_hop[~valid] = replacement
    np.fill_diagonal(best_two_hop, replacement)

    direct = clean
    advantage = np.maximum(best_two_hop - direct, 0.0)
    bridge = 1.0 - np.exp(
        -np.minimum(advantage / max(scale, eps), 30.0)
    )
    bridge[~valid] = 0.0
    np.fill_diagonal(bridge, 0.0)

    inverse_max = float(np.max(inverse))
    if np.isfinite(inverse_max) and inverse_max > 0.0:
        distance_affinity = inverse / inverse_max
    else:
        distance_affinity = valid.astype(float)

    score = distance_affinity
    score *= 0.55 + 0.45 * competition
    score *= 0.72 + 0.28 * mutual_support
    score *= 0.78 + 0.22 * np.minimum(endpoint_scarcity, 2.5) / 2.5
    score *= 0.82 + 0.18 * bridge

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
