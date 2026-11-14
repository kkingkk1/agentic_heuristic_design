import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
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
    distances = np.where(np.isfinite(cleaned), cleaned, cleaned.T)
    distances = np.where(np.isfinite(distances), distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    base_scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(base_scale) or base_scale <= 0.0:
        base_scale = 1.0

    maximum = float(np.max(observed)) if observed.size else base_scale
    missing_penalty = maximum + 32.0 * base_scale

    distances = np.where(valid, distances, missing_penalty)
    distances[diagonal] = missing_penalty

    positive_valid = distances[valid]
    global_temperature = (
        float(np.median(positive_valid))
        if positive_valid.size
        else base_scale
    )
    global_temperature = max(global_temperature, np.finfo(float).eps)

    affinity = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        row_distances = distances[i, row_valid]
        local_scale = float(np.median(row_distances))
        local_scale = max(
            0.5 * local_scale,
            0.25 * global_temperature,
            np.finfo(float).eps,
        )

        affinity[i, row_valid] = np.exp(
            -np.clip(
                (row_distances - np.min(row_distances)) / local_scale,
                0.0,
                40.0,
            )
        )

    affinity = 0.5 * (affinity + affinity.T)
    maximum_affinity = float(np.max(affinity))
    if maximum_affinity > 0.0 and np.isfinite(maximum_affinity):
        affinity /= maximum_affinity

    sorted_distances = np.sort(
        np.where(valid, distances, missing_penalty),
        axis=1,
    )

    nearest = sorted_distances[:, 0]
    second_nearest = (
        sorted_distances[:, 1]
        if n > 2
        else sorted_distances[:, 0]
    )

    regret = np.maximum(second_nearest - nearest, 0.0)
    positive_regret = regret[regret > 0.0]
    regret_scale = (
        float(np.median(positive_regret))
        if positive_regret.size
        else base_scale
    )
    regret_scale = max(regret_scale, np.finfo(float).eps)

    scarcity = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        row_distances = distances[i]
        local_best = nearest[i]
        local_second = second_nearest[i]

        gap = np.maximum(local_second - row_distances, 0.0)
        gap[row_distances <= local_best] = np.maximum(
            local_second - row_distances[row_distances <= local_best],
            0.0,
        )

        scarcity[i, row_valid] = 1.0 - np.exp(
            -np.clip(gap[row_valid] / regret_scale, 0.0, 30.0)
        )

    scarcity = np.sqrt(
        np.maximum(scarcity, 0.0)
        * np.maximum(scarcity.T, 0.0)
    )

    two_hop = np.full((n, n), missing_penalty, dtype=float)

    for k in range(n):
        left = distances[:, k][:, None]
        right = distances[k, :][None, :]
        candidate = left + right

        feasible = valid[:, k][:, None] & valid[k, :][None, :]
        candidate = np.where(feasible, candidate, missing_penalty)
        two_hop = np.minimum(two_hop, candidate)

    two_hop[diagonal] = missing_penalty

    bypass_gain = np.maximum(two_hop - distances, 0.0)
    bridge_scale = max(base_scale, np.finfo(float).eps)
    bridge = 1.0 - np.exp(
        -np.clip(bypass_gain / bridge_scale, 0.0, 30.0)
    )
    bridge[~valid] = 0.0

    score = (
        0.52 * affinity
        + 0.28 * scarcity
        + 0.20 * bridge
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    epsilon = np.finfo(float).eps

    for _ in range(16):
        row_sum = balanced.sum(axis=1, keepdims=True)
        active = row_sum[:, 0] > epsilon
        balanced[active] /= row_sum[active]
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if balanced_max > 0.0 and np.isfinite(balanced_max):
        balanced /= balanced_max

    result = 0.70 * score + 0.30 * balanced
    result[~valid] = 0.0
    result[diagonal] = 0.0

    maximum_result = float(np.max(result))
    if maximum_result > 0.0 and np.isfinite(maximum_result):
        result /= maximum_result

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
