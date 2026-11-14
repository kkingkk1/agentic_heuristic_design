import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-selection priors for TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(np.isfinite(raw) & (raw >= 0.0), raw, 0.0)
    distances = 0.5 * (cleaned + cleaned.T)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    upper = float(np.quantile(observed, 0.90)) if observed.size else scale
    penalty = max(upper, scale) + 10.0 * scale
    distances = np.where(valid, distances, penalty)
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps * max(scale, 1.0)
    temperature = max(0.55 * scale, eps)

    affinity = np.exp(-np.minimum(distances / temperature, 60.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_max = np.maximum(np.max(affinity, axis=1, keepdims=True), eps)
    col_max = np.maximum(np.max(affinity, axis=0, keepdims=True), eps)

    row_preference = affinity / row_max
    col_preference = affinity / col_max
    mutual_preference = np.sqrt(row_preference * col_preference)

    row_sum = np.maximum(affinity.sum(axis=1, keepdims=True), eps)
    col_sum = np.maximum(affinity.sum(axis=0, keepdims=True), eps)

    row_share = affinity / row_sum
    col_share = affinity / col_sum
    competition = np.sqrt(row_share * col_share)

    endpoint_gap = np.zeros((n, n), dtype=float)
    local_floor = max(0.20 * scale, eps)

    for i in range(n):
        candidates = distances[i].copy()
        candidates[i] = penalty
        order = np.argsort(candidates, kind="stable")
        best = candidates[order[0]]
        second = candidates[order[1]] if n > 2 else best
        local_scale = max(second - best, local_floor)

        gap = best - distances[i]
        endpoint_gap[i] = np.exp(
            np.clip(gap / local_scale, -12.0, 12.0)
        )
        endpoint_gap[i, ~valid[i]] = 0.0

    endpoint_exclusivity = np.sqrt(endpoint_gap * endpoint_gap.T)
    endpoint_exclusivity[~valid] = 0.0

    bypass = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        candidate = distances[:, k, None] + distances[None, :, k]
        bypass = np.minimum(bypass, candidate)

    np.fill_diagonal(bypass, penalty)
    advantage = np.maximum(bypass - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -advantage / max(0.80 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    base = affinity / max(float(np.max(affinity)), eps)

    score = (
        0.42 * base
        + 0.23 * mutual_preference
        + 0.20 * endpoint_exclusivity
        + 0.15 * irreplaceability
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(32):
        row_total = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            eps,
        )
        balanced /= row_total
        balanced = 0.5 * (balanced + balanced.T)

    result = 0.62 * score + 0.38 * balanced
    result[~valid] = 0.0
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
