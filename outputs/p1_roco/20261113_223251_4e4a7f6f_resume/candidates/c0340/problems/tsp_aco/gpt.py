import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & offdiag

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(np.isfinite(raw), raw, np.nan)
    cleaned = np.where(np.isfinite(cleaned), np.maximum(cleaned, 0.0), np.nan)

    observed = cleaned[finite]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_max = float(np.nanmax(observed)) if observed.size else scale
    penalty = max(finite_max + 12.0 * scale, 12.0 * scale)

    distances = np.where(np.isfinite(cleaned), cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    valid = distances[offdiag]
    valid = valid[np.isfinite(valid) & (valid > 0.0)]
    temperature = float(np.median(valid)) if valid.size else scale
    temperature = max(0.24 * temperature, np.finfo(float).eps)

    affinity = np.exp(-np.minimum(distances, penalty) / temperature)
    affinity[~finite] = 0.0
    np.fill_diagonal(affinity, 0.0)

    if not np.any(affinity > 0.0):
        result = np.ones((n, n), dtype=float)
        result[~finite] = 0.0
        np.fill_diagonal(result, 0.0)
        return result

    row_sorted = np.sort(distances, axis=1)
    best = row_sorted[:, 0]
    second = row_sorted[:, 1] if n > 1 else best
    regret_scale = np.maximum(second - best, 0.20 * scale)
    regret_scale = np.maximum(regret_scale, np.finfo(float).eps)

    endpoint = np.zeros((n, n), dtype=float)
    for i in range(n):
        improvement = distances[i, :] - best[i]
        endpoint[i, :] = np.exp(
            -np.clip(improvement / regret_scale[i], 0.0, 24.0)
        )

    endpoint[~finite] = 0.0
    endpoint = np.sqrt(endpoint * endpoint.T)

    two_hop = distances[:, :, None] + distances[None, :, :]
    bypass = np.min(two_hop, axis=1)
    np.fill_diagonal(bypass, penalty)

    excess = np.maximum(bypass - distances, 0.0)
    bypass_scale = max(0.65 * scale, np.finfo(float).eps)
    irreplaceability = 1.0 - np.exp(-excess / bypass_scale)
    irreplaceability[~finite] = 0.0

    neighborhood = np.zeros((n, n), dtype=float)
    k = min(max(2, int(np.sqrt(n))), n - 1)

    for i in range(n):
        row = distances[i].copy()
        row[i] = penalty
        nearest = np.partition(row, k - 1)[:k]
        local = max(float(np.mean(nearest)), 0.25 * scale)
        neighborhood[i] = np.exp(-row / local)

    neighborhood[~finite] = 0.0
    neighborhood = np.sqrt(neighborhood * neighborhood.T)

    affinity /= max(float(np.max(affinity)), np.finfo(float).eps)

    score = (
        0.50 * affinity
        + 0.23 * endpoint
        + 0.19 * irreplaceability
        + 0.08 * neighborhood
    )
    score[~finite] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(32):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, np.finfo(float).eps)
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.62 * score + 0.38 * balanced
    result[~finite] = 0.0
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
