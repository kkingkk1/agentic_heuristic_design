import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric normalized edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid = np.isfinite(raw) & (raw >= 0.0) & ~diagonal
    symmetric_valid = valid & valid.T

    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    distances = np.full((n, n), np.inf, dtype=float)
    distances[symmetric_valid] = 0.5 * (
        raw[symmetric_valid] + raw.T[symmetric_valid]
    )
    np.fill_diagonal(distances, np.inf)

    finite_edges = distances[symmetric_valid]
    positive_edges = finite_edges[finite_edges > 0.0]

    if positive_edges.size:
        scale = float(np.median(positive_edges))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)

    reciprocal = np.zeros_like(distances)
    reciprocal[symmetric_valid] = 1.0 / (
        distances[symmetric_valid] + eps
    )

    direct = reciprocal / max(float(np.max(reciprocal)), eps)

    row_peak = np.maximum(
        np.max(reciprocal, axis=1, keepdims=True),
        eps,
    )
    col_peak = np.maximum(
        np.max(reciprocal, axis=0, keepdims=True),
        eps,
    )

    row_competitiveness = reciprocal / row_peak
    col_competitiveness = reciprocal / col_peak
    mutual_competitiveness = np.sqrt(
        np.maximum(row_competitiveness * col_competitiveness, 0.0)
    )

    proximity = np.zeros_like(distances)
    finite_mask = symmetric_valid
    proximity[finite_mask] = np.exp(
        -np.minimum(distances[finite_mask] / scale, 60.0)
    )

    row_mass = np.maximum(
        np.sum(proximity, axis=1),
        eps,
    )
    local_preference = proximity / row_mass[:, None]
    mutual_support = np.sqrt(
        np.maximum(local_preference * local_preference.T, 0.0)
    )

    scarcity = 1.0 / row_mass
    scarcity /= max(float(np.mean(scarcity)), eps)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )
    endpoint_scarcity = np.clip(endpoint_scarcity, 0.0, 2.0)

    bypass = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        candidate = distances[:, k, None] + distances[None, k, :]
        bypass = np.minimum(bypass, candidate)

    bypass_advantage = np.zeros_like(distances)
    finite_bypass = np.isfinite(bypass) & symmetric_valid

    bypass_advantage[finite_bypass] = np.maximum(
        bypass[finite_bypass] - distances[finite_bypass],
        0.0,
    )

    irreplaceability = np.zeros_like(distances)
    irreplaceability[finite_bypass] = 1.0 - np.exp(
        -np.minimum(
            bypass_advantage[finite_bypass] / scale,
            60.0,
        )
    )

    no_bypass = symmetric_valid & ~np.isfinite(bypass)
    irreplaceability[no_bypass] = 1.0

    local_gain = (
        0.50 * mutual_competitiveness
        + 0.30 * mutual_support
        + 0.20 * endpoint_scarcity
    )

    score = direct * (0.65 + 0.35 * local_gain)
    score += 0.25 * irreplaceability * (
        0.5 + 0.5 * mutual_competitiveness
    )

    score[~symmetric_valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()

    for _ in range(32):
        row_sum = np.sum(balanced, axis=1, keepdims=True)
        balanced /= np.maximum(row_sum, eps)
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~symmetric_valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    result = 0.70 * score + 0.30 * balanced
    result[~symmetric_valid] = 0.0
    result = 0.5 * (result + result.T)
    result = np.maximum(result, 0.0)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > 0.0:
        result /= maximum
    else:
        result.fill(0.0)

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
