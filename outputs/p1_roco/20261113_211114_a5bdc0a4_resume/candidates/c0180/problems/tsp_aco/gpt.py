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

    nonnegative = np.where(finite, np.maximum(raw, 0.0), np.nan)
    distances = np.where(
        np.isfinite(nonnegative),
        nonnegative,
        nonnegative.T,
    )
    distances = np.where(np.isfinite(distances), distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    observed = distances[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    maximum = float(np.max(observed)) if observed.size else scale
    penalty = maximum + 20.0 * scale
    distances = np.where(valid, distances, penalty)
    np.fill_diagonal(distances, penalty)

    usable = distances[valid]
    temperature = float(np.median(usable)) if usable.size else scale
    temperature = max(0.42 * temperature, 0.5 * scale, np.finfo(float).eps)

    affinity = np.exp(-np.minimum(distances, penalty) / temperature)
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    row_sorted = np.sort(
        np.where(valid, distances, penalty),
        axis=1,
    )

    best = row_sorted[:, 0]
    second = row_sorted[:, 1] if n > 2 else best
    local_regret = np.maximum(second - best, 0.0)

    positive_regret = local_regret[local_regret > 0.0]
    regret_scale = (
        float(np.median(positive_regret))
        if positive_regret.size
        else scale
    )
    regret_scale = max(0.8 * regret_scale, 0.25 * scale)

    endpoint_strength = np.zeros((n, n), dtype=float)
    for i in range(n):
        alternatives = np.where(valid[i], distances[i], penalty)
        alternatives[i] = penalty

        order = np.argsort(alternatives, kind="stable")
        first = alternatives[order[0]]
        second_best = alternatives[order[1]] if n > 2 else first

        gap = np.where(
            np.arange(n) == order[0],
            second_best - alternatives,
            first - alternatives,
        )

        endpoint_strength[i] = np.exp(
            np.clip(gap / regret_scale, -18.0, 18.0)
        )

    endpoint_strength[~valid] = 0.0
    endpoint_strength = np.sqrt(
        np.maximum(endpoint_strength, 0.0)
        * np.maximum(endpoint_strength.T, 0.0)
    )

    bypass = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        candidate = distances[:, k, None] + distances[None, k, :]
        candidate[~valid] = penalty
        bypass = np.minimum(bypass, candidate)

    bypass[diagonal] = penalty
    bypass_advantage = np.maximum(bypass - distances, 0.0)

    bridge = 1.0 - np.exp(
        -bypass_advantage / max(0.85 * scale, np.finfo(float).eps)
    )
    bridge[~valid] = 0.0

    affinity_max = float(np.max(affinity))
    if np.isfinite(affinity_max) and affinity_max > 0.0:
        affinity /= affinity_max

    endpoint_max = float(np.max(endpoint_strength))
    if np.isfinite(endpoint_max) and endpoint_max > 0.0:
        endpoint_strength /= endpoint_max

    score = (
        0.56 * affinity
        + 0.26 * endpoint_strength
        + 0.18 * bridge
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    balanced = score.copy()
    for _ in range(16):
        row_sum = balanced.sum(axis=1, keepdims=True)
        active = row_sum[:, 0] > np.finfo(float).eps
        balanced[active] /= row_sum[active]
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced *= balanced_max

    result = 0.78 * score + 0.22 * balanced
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
