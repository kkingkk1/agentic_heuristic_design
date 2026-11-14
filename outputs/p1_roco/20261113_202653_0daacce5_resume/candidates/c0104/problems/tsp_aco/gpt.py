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

    d = np.where(np.isfinite(raw), np.maximum(raw, 0.0), np.inf)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d)
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = d[valid & (d > 0.0)]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps * max(global_scale, 1.0)

    reciprocal = np.zeros((n, n), dtype=float)
    reciprocal[valid] = 1.0 / (d[valid] + eps)

    row_peak = np.maximum(
        np.max(reciprocal, axis=1, keepdims=True),
        eps,
    )
    local_competition = reciprocal / row_peak
    mutual_competition = np.sqrt(
        np.maximum(local_competition * local_competition.T, 0.0)
    )

    local_scale = np.empty(n, dtype=float)
    neighborhood_size = min(4, n - 1)

    for i in range(n):
        neighbors = d[i, valid[i]]
        if neighbors.size == 0:
            local_scale[i] = global_scale
            continue

        nearest = np.sort(neighbors)[:neighborhood_size]
        positive_nearest = nearest[nearest > 0.0]
        if positive_nearest.size:
            local_scale[i] = float(np.median(positive_nearest))
        else:
            local_scale[i] = global_scale

    local_scale = np.maximum(local_scale, eps)

    soft_preference = np.zeros((n, n), dtype=float)
    soft_preference[valid] = np.exp(
        -d[valid] / local_scale[:, None][valid]
    )

    soft_mass = np.maximum(
        soft_preference.sum(axis=1, keepdims=True),
        eps,
    )
    normalized_preference = soft_preference / soft_mass
    mutual_preference = np.sqrt(
        np.maximum(
            normalized_preference * normalized_preference.T,
            0.0,
        )
    )

    local_density = 1.0 / np.maximum(local_scale, eps)
    local_density /= max(float(np.max(local_density)), eps)

    density_support = np.sqrt(
        np.maximum(
            local_density[:, None] * local_density[None, :],
            0.0,
        )
    )

    row_best = np.max(np.where(valid, d, np.inf), axis=1)
    row_best = np.min(np.where(valid, d, np.inf), axis=1)

    alternative = np.where(valid, d, np.inf)
    alternative_gap = np.zeros((n, n), dtype=float)

    for i in range(n):
        candidates = alternative[i].copy()
        finite_candidates = candidates[np.isfinite(candidates)]
        if finite_candidates.size == 0:
            continue

        best = float(np.min(finite_candidates))
        second = (
            float(np.partition(finite_candidates, 1)[1])
            if finite_candidates.size > 1
            else best
        )

        for j in range(n):
            if not valid[i, j]:
                continue
            reference = second if d[i, j] <= best else best
            alternative_gap[i, j] = max(reference - d[i, j], 0.0)

    gap_scale = max(global_scale, eps)
    exclusivity = 1.0 - np.exp(-alternative_gap / gap_scale)
    exclusivity = np.sqrt(
        np.maximum(exclusivity * exclusivity.T, 0.0)
    )

    safe_d = np.where(valid, d, np.inf)
    bypass = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        candidate = safe_d[:, k, None] + safe_d[None, k, :]
        bypass = np.minimum(bypass, candidate)

    finite_bypass = np.isfinite(bypass)
    advantage = np.zeros((n, n), dtype=float)
    advantage[valid & finite_bypass] = np.maximum(
        bypass[valid & finite_bypass] - d[valid & finite_bypass],
        0.0,
    )
    advantage[valid & ~finite_bypass] = global_scale

    irreplaceability = 1.0 - np.exp(-advantage / global_scale)

    base = reciprocal / max(float(np.max(reciprocal)), eps)

    score = base
    score *= 0.65 + 0.35 * mutual_preference
    score *= 0.80 + 0.20 * density_support
    score *= 0.75 + 0.25 * exclusivity
    score *= 0.80 + 0.20 * irreplaceability

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()
    for _ in range(12):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            eps,
        )
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)

    score = 0.70 * score + 0.30 * balanced
    score[~valid] = 0.0
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
    )
