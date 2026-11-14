import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-selection priors for TSP sampling."""
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
    cleaned = np.where(
        np.isfinite(cleaned),
        cleaned,
        np.where(np.isfinite(cleaned.T), cleaned.T, np.nan),
    )

    observed = cleaned[valid]
    positive = observed[observed > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    fallback = float(np.max(observed[np.isfinite(observed)]))
    if not np.isfinite(fallback):
        fallback = scale
    penalty = fallback + 16.0 * scale

    distances = np.where(np.isfinite(cleaned), cleaned, penalty)
    distances = 0.5 * (distances + distances.T)
    distances[~valid] = penalty
    np.fill_diagonal(distances, penalty)

    eps = np.finfo(float).eps
    temperature = max(
        scale,
        float(np.median(distances[valid])) if np.any(valid) else scale,
    )

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(
        -np.minimum(distances[valid] / temperature, 40.0)
    )

    maximum = float(np.max(affinity))
    if maximum > eps and np.isfinite(maximum):
        affinity /= maximum

    row_mass = np.maximum(affinity.sum(axis=1, keepdims=True), eps)
    col_mass = np.maximum(affinity.sum(axis=0, keepdims=True), eps)
    competition = np.sqrt(
        (affinity / row_mass) * (affinity / col_mass)
    )
    competition[~valid] = 0.0

    sorted_distances = np.sort(
        np.where(valid, distances, penalty),
        axis=1,
    )
    best = sorted_distances[:, 0]
    second = sorted_distances[:, min(1, n - 1)]
    regret = np.maximum(second - best, 0.0)

    positive_regret = regret[regret > eps]
    regret_scale = (
        float(np.median(positive_regret))
        if positive_regret.size
        else scale
    )
    regret_scale = max(regret_scale, eps)

    row_regret = np.maximum(
        sorted_distances[:, min(1, n - 1), None] - distances,
        0.0,
    )
    col_distances = distances.T
    sorted_col = np.sort(
        np.where(valid.T, col_distances, penalty),
        axis=1,
    )
    col_regret = np.maximum(
        sorted_col[:, min(1, n - 1), None] - col_distances,
        0.0,
    ).T

    endpoint = 1.0 - np.exp(
        -0.5 * (row_regret + col_regret) / regret_scale
    )
    endpoint[~valid] = 0.0

    bridge = np.full((n, n), penalty, dtype=float)
    for k in range(n):
        via_k = distances[:, k, None] + distances[None, k, :]
        bridge = np.minimum(bridge, via_k)

    bridge_advantage = np.maximum(bridge - distances, 0.0)
    irreplaceability = 1.0 - np.exp(
        -bridge_advantage / max(scale, eps)
    )
    irreplaceability[~valid] = 0.0

    score = (
        0.46 * affinity
        + 0.24 * competition
        + 0.18 * endpoint
        + 0.12 * irreplaceability
    )
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)

    exploration = 0.06
    valid_count = float(np.sum(valid))
    prior = (
        (1.0 - exploration) * score
        + exploration * valid.astype(float) / max(valid_count, 1.0)
    )
    prior[~valid] = 0.0

    balanced = prior.copy()
    for _ in range(16):
        row_sum = balanced.sum(axis=1, keepdims=True)
        active = row_sum[:, 0] > eps
        balanced[active] /= row_sum[active, None]
        balanced = 0.5 * (balanced + balanced.T)
        balanced[~valid] = 0.0

    result = 0.72 * prior + 0.28 * balanced
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
