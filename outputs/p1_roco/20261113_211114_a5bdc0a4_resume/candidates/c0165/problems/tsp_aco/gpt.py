import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)
    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~diagonal

    if not np.any(valid):
        return result

    cleaned = np.where(finite, np.maximum(raw, 0.0), np.nan)
    distances = np.where(
        np.isfinite(cleaned),
        cleaned,
        cleaned.T,
    )
    distances = np.where(np.isfinite(distances), distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else max(float(np.median(observed)), 1.0)
    )
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    maximum = float(np.max(observed)) if observed.size else global_scale
    penalty = maximum + 32.0 * global_scale
    filled = np.where(valid, distances, penalty)
    np.fill_diagonal(filled, penalty)

    eps = np.finfo(float).eps

    local_scales = np.empty(n, dtype=float)
    nearest = np.empty(n, dtype=float)
    second_nearest = np.empty(n, dtype=float)

    for i in range(n):
        row = filled[i, valid[i]]
        if row.size == 0:
            local_scales[i] = global_scale
            nearest[i] = global_scale
            second_nearest[i] = global_scale
            continue

        ordered = np.sort(row)
        nearest[i] = ordered[0]
        second_nearest[i] = ordered[1] if ordered.size > 1 else ordered[0]

        local_positive = ordered[ordered > 0.0]
        scale = (
            float(np.median(local_positive))
            if local_positive.size
            else global_scale
        )
        local_scales[i] = max(scale, 0.25 * global_scale, eps)

    local_scales = np.maximum(local_scales, eps)

    row_scale = 0.5 * (local_scales[:, None] + local_scales[None, :])
    row_scale = np.maximum(row_scale, eps)

    normalized_distance = filled / row_scale
    affinity = np.exp(-np.clip(normalized_distance, 0.0, 40.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    rank_weight = np.zeros((n, n), dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size == 0:
            continue

        order = candidates[np.argsort(filled[i, candidates], kind="stable")]
        ranks = np.arange(order.size, dtype=float)
        rank_weight[i, order] = np.exp(-ranks / max(1.5, 0.35 * order.size))

    rank_weight = 0.5 * (rank_weight + rank_weight.T)

    endpoint_gap_i = np.maximum(second_nearest[:, None] - filled, 0.0)
    endpoint_gap_j = np.maximum(second_nearest[None, :] - filled, 0.0)

    endpoint_regret = 1.0 - np.exp(
        -0.5
        * (
            endpoint_gap_i / local_scales[:, None]
            + endpoint_gap_j / local_scales[None, :]
        )
    )
    endpoint_regret[~valid] = 0.0
    np.fill_diagonal(endpoint_regret, 0.0)

    bypass = np.full((n, n), penalty, dtype=float)

    for k in range(n):
        left = valid[:, k]
        right = valid[k, :]
        candidate = filled[:, k, None] + filled[None, k, :]
        allowed = left[:, None] & right[None, :]
        allowed &= ~diagonal
        allowed &= ~np.equal(np.arange(n)[:, None], k)
        allowed &= ~np.equal(np.arange(n)[None, :], k)
        candidate = np.where(allowed, candidate, penalty)
        bypass = np.minimum(bypass, candidate)

    bypass[diagonal] = penalty

    detour_scale = np.maximum(row_scale, 0.5 * global_scale)
    bypass_advantage = np.maximum(bypass - filled, 0.0)
    bridge = 1.0 - np.exp(
        -bypass_advantage / np.maximum(detour_scale, eps)
    )
    bridge[~valid] = 0.0
    np.fill_diagonal(bridge, 0.0)

    affinity /= max(float(np.max(affinity)), eps)
    rank_weight /= max(float(np.max(rank_weight)), eps)
    endpoint_regret /= max(float(np.max(endpoint_regret)), eps)
    bridge /= max(float(np.max(bridge)), eps)

    score = (
        0.42 * affinity
        + 0.23 * rank_weight
        + 0.23 * endpoint_regret
        + 0.12 * bridge
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(12):
        row_sum = balanced.sum(axis=1, keepdims=True)
        active = row_sum[:, 0] > eps
        balanced[active] /= row_sum[active]
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    if np.max(balanced) > eps:
        balanced /= np.max(balanced)

    result = 0.78 * score + 0.22 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if np.isfinite(maximum) and maximum > eps:
        result /= maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
