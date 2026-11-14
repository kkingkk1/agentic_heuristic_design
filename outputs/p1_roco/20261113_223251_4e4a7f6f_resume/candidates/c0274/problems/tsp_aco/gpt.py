import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)
    both = finite & finite.T
    only_forward = finite & ~finite.T
    only_backward = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[only_forward] = raw[only_forward]
    cleaned[only_backward] = raw.T[only_backward]
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    largest = float(np.max(observed)) if observed.size else scale
    penalty = largest + 8.0 * scale

    distances = np.full((n, n), penalty, dtype=float)
    distances[valid] = cleaned[valid]
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, penalty)

    temperature = max(0.28 * scale, eps)
    affinity = np.exp(-np.minimum(distances / temperature, 60.0))
    affinity[~valid] = 0.0
    np.fill_diagonal(affinity, 0.0)

    maximum_affinity = float(np.max(affinity))
    if not np.isfinite(maximum_affinity) or maximum_affinity <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    base = affinity / maximum_affinity

    ranked = np.sort(distances, axis=1)
    nearest = ranked[:, 0]
    second = ranked[:, 1] if n > 1 else nearest
    local_scale = np.maximum(second - nearest, 0.18 * scale)
    local_scale = np.maximum(local_scale, eps)

    local_preference = np.exp(
        np.clip(
            -(distances - nearest[:, None]) / local_scale[:, None],
            -24.0,
            0.0,
        )
    )
    local_preference[~valid] = 0.0
    np.fill_diagonal(local_preference, 0.0)

    preference_max = np.maximum(
        np.max(local_preference, axis=1, keepdims=True),
        eps,
    )
    local_preference /= preference_max

    endpoint_competition = np.sqrt(
        np.maximum(local_preference, 0.0)
        * np.maximum(local_preference.T, 0.0)
    )

    regret = np.exp(
        np.clip(
            (second[:, None] - distances) / local_scale[:, None],
            -20.0,
            20.0,
        )
    )
    regret[~valid] = 0.0
    np.fill_diagonal(regret, 0.0)

    endpoint_regret = np.sqrt(
        np.maximum(regret, 0.0) * np.maximum(regret.T, 0.0)
    )

    best_two_hop = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        via_k = distances[:, k, None] + distances[None, k, :]
        best_two_hop = np.minimum(best_two_hop, via_k)

    np.fill_diagonal(best_two_hop, penalty)

    bypass_gain = np.maximum(best_two_hop - distances, 0.0)
    bypass_scale = max(0.75 * scale, eps)
    irreplaceability = 1.0 - np.exp(-bypass_gain / bypass_scale)
    irreplaceability[~valid] = 0.0
    np.fill_diagonal(irreplaceability, 0.0)

    score = (
        0.48 * base
        + 0.24 * endpoint_competition
        + 0.14 * endpoint_regret
        + 0.14 * irreplaceability
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    balanced = score.copy()
    for _ in range(40):
        row_sum = np.maximum(
            balanced.sum(axis=1, keepdims=True),
            eps,
        )
        balanced /= row_sum
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0
        np.fill_diagonal(balanced, 0.0)

    balanced_max = float(np.max(balanced))
    if np.isfinite(balanced_max) and balanced_max > 0.0:
        balanced /= balanced_max

    result = 0.62 * score + 0.38 * balanced
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
    ).astype(float, copy=False)
