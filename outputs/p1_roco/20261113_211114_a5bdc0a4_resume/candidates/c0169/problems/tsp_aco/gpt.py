import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.where(finite, np.maximum(raw, 0.0), np.nan)
    distances = np.where(np.isfinite(cleaned), cleaned, cleaned.T)
    distances = np.where(np.isfinite(distances), distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    penalty = max(maximum + 32.0 * scale, scale)
    distances = np.where(valid, distances, penalty)
    np.fill_diagonal(distances, penalty)

    usable = distances[valid]
    median_distance = (
        float(np.median(usable)) if usable.size else scale
    )
    temperature = max(
        0.35 * median_distance,
        0.25 * scale,
        np.finfo(float).eps,
    )

    positive_distances = np.where(valid, distances, np.inf)
    nearest = np.min(positive_distances, axis=1)
    second_nearest = np.partition(
        positive_distances,
        kth=min(1, n - 1),
        axis=1,
    )[:, min(1, n - 1)]

    finite_second = np.isfinite(second_nearest)
    second_nearest = np.where(finite_second, second_nearest, nearest)
    regret = np.maximum(second_nearest - nearest, 0.0)

    positive_regret = regret[regret > 0.0]
    regret_scale = (
        float(np.median(positive_regret))
        if positive_regret.size
        else scale
    )
    regret_scale = max(regret_scale, 0.25 * scale, np.finfo(float).eps)

    closeness = np.exp(
        -np.minimum(distances, penalty) / temperature
    )
    closeness[~valid] = 0.0
    closeness /= max(
        float(np.max(closeness)),
        np.finfo(float).eps,
    )

    endpoint_exclusivity = np.exp(
        np.clip(
            (second_nearest[:, None] - distances) / regret_scale,
            -20.0,
            20.0,
        )
    )
    endpoint_exclusivity[~valid] = 0.0
    endpoint_exclusivity = np.sqrt(
        np.maximum(endpoint_exclusivity, 0.0)
        * np.maximum(endpoint_exclusivity.T, 0.0)
    )
    endpoint_exclusivity /= max(
        float(np.max(endpoint_exclusivity)),
        np.finfo(float).eps,
    )

    rank_costs = np.where(valid, distances, penalty)
    rank_order = np.argsort(rank_costs, axis=1, kind="stable")
    ranks = np.empty_like(rank_order, dtype=float)
    row_index = np.arange(n)[:, None]
    ranks[row_index, rank_order] = np.arange(n, dtype=float)
    rank_affinity = 1.0 / (1.0 + ranks)
    rank_affinity[~valid] = 0.0

    mutuality = np.sqrt(
        np.maximum(rank_affinity, 0.0)
        * np.maximum(rank_affinity.T, 0.0)
    )
    mutuality /= max(
        float(np.max(mutuality)),
        np.finfo(float).eps,
    )

    bypass = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        left = distances[:, k][:, None]
        right = distances[k, :][None, :]
        candidate = left + right
        candidate[~valid] = penalty
        candidate[diagonal] = penalty
        bypass = np.minimum(bypass, candidate)

    bypass_advantage = np.maximum(
        bypass - np.minimum(distances, penalty),
        0.0,
    )
    bridge_scale = max(scale, 0.5 * median_distance)
    bridge = 1.0 - np.exp(-bypass_advantage / bridge_scale)
    bridge[~valid] = 0.0
    bridge /= max(
        float(np.max(bridge)),
        np.finfo(float).eps,
    )

    score = (
        0.42 * closeness
        + 0.25 * endpoint_exclusivity
        + 0.18 * mutuality
        + 0.15 * bridge
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(20):
        row_sum = balanced.sum(axis=1, keepdims=True)
        active = row_sum[:, 0] > np.finfo(float).eps
        balanced[active] /= row_sum[active]
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.78 * score + 0.22 * balanced
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
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
