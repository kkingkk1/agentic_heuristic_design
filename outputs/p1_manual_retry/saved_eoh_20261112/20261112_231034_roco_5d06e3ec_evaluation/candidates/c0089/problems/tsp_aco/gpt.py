import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-priority matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
    if np.any(np.isnan(raw)) or np.any(raw < 0):
        raise ValueError("distance_matrix must contain nonnegative, non-NaN values")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & ~diagonal

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    # Build a symmetric distance representation while preserving one-sided
    # finite entries when the input contains missing directed distances.
    reverse = raw.T
    both_finite = np.isfinite(raw) & np.isfinite(reverse)
    only_forward = np.isfinite(raw) & ~np.isfinite(reverse)
    only_reverse = ~np.isfinite(raw) & np.isfinite(reverse)

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_finite] = 0.5 * (raw[both_finite] + reverse[both_finite])
    distance[only_forward] = raw[only_forward]
    distance[only_reverse] = reverse[only_reverse]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    values = distance[valid]

    eps = np.finfo(float).eps
    center = float(np.median(values)) if values.size else 1.0
    spread = float(np.percentile(values, 75.0) - np.percentile(values, 25.0))
    scale = max(spread, center, eps)

    # Global distance compression. This does not depend on endpoint-local
    # neighborhoods or rank assignments.
    normalized = np.zeros_like(distance)
    normalized[valid] = distance[valid] / scale
    distance_preference = np.zeros_like(distance)
    distance_preference[valid] = np.exp(
        -0.5 * np.minimum(normalized[valid], 50.0) ** 2
    )

    # Estimate the best two-hop bypass for every edge.
    # The intermediate node must differ from both endpoints.
    bypass = np.full((n, n), np.inf, dtype=float)
    for k in range(n):
        left = distance[:, k][:, None]
        right = distance[k, :][None, :]
        candidate = left + right
        candidate[k, :] = np.inf
        candidate[:, k] = np.inf
        bypass = np.minimum(bypass, candidate)

    # Positive detour savings indicate that the direct edge is valuable.
    savings = np.zeros_like(distance)
    usable_bypass = valid & np.isfinite(bypass)
    savings[usable_bypass] = bypass[usable_bypass] - distance[usable_bypass]

    scaled_savings = np.clip(savings / scale, -50.0, 50.0)
    detour_value = np.zeros_like(distance)
    detour_value[valid] = 1.0 / (1.0 + np.exp(-scaled_savings[valid]))

    # If no two-hop bypass exists, retain a neutral contribution.
    detour_value[valid & ~np.isfinite(bypass)] = 0.5

    score = distance_preference * (0.25 + 0.75 * detour_value)
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    positive = score > 0.0
    score[positive] += eps

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
