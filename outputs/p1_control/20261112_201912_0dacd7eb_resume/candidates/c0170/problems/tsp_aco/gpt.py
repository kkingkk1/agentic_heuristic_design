import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0.0)
    np.fill_diagonal(positive, False)

    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[positive]
    replacement = float(np.max(positive_values))
    if not np.isfinite(replacement) or replacement <= 0.0:
        replacement = 1.0

    d = np.where(finite, distances, replacement)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    replacement = float(np.max(finite_values))
    if not np.isfinite(replacement) or replacement <= 0.0:
        replacement = 1.0

    d = np.where(np.isfinite(d), d, replacement)
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    valid_count = max(1, n - 1)
    candidate_count = max(2, min(valid_count, int(np.sqrt(n)) + 2))

    ordered = np.sort(d, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, min(1, valid_count - 1)]
    third = ordered[:, min(2, valid_count - 1)]

    local_k = min(candidate_count, valid_count)
    local_values = ordered[:, :local_k]
    local_weights = np.linspace(1.0, 0.55, local_k)
    local_scale = (
        local_values @ local_weights / np.sum(local_weights)
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])
    normalized = d / np.maximum(pair_scale, eps)

    endpoint_i = np.exp(-np.clip(normalized, 0.0, 40.0))
    endpoint_j = endpoint_i.T
    local_affinity = np.sqrt(endpoint_i * endpoint_j)

    global_values = d[np.isfinite(d)]
    global_scale = float(np.median(global_values))
    global_scale = max(global_scale, eps)
    global_affinity = np.exp(
        -np.clip((d / global_scale) ** 0.75, 0.0, 30.0)
    )

    rank_i = np.argsort(np.argsort(d, axis=1, kind="stable"), axis=1)
    rank_j = rank_i.T
    rank_scale = max(float(valid_count), 1.0)
    rank_affinity = np.exp(
        -0.7 * (rank_i + rank_j) / rank_scale
    )

    nearest_gap = np.maximum(second - nearest, eps)
    local_regret = np.clip((d - nearest[:, None]) / nearest_gap[:, None], 0.0, 6.0)
    local_regret_t = local_regret.T
    regret_penalty = np.exp(-0.18 * (local_regret + local_regret_t))

    nearest_mask_i = d <= (nearest[:, None] + 0.05 * nearest_gap[:, None])
    nearest_mask_j = nearest_mask_i.T
    mutual_candidate = nearest_mask_i.astype(float) * nearest_mask_j.astype(float)

    relative_second_i = np.clip(
        (second[:, None] - d) / np.maximum(second[:, None], eps),
        0.0,
        1.0,
    )
    relative_second_j = relative_second_i.T
    two_sided_advantage = np.sqrt(relative_second_i * relative_second_j)

    scarcity_i = np.clip(
        (third[:, None] - d) / np.maximum(third[:, None] - nearest[:, None], eps),
        0.0,
        1.0,
    )
    scarcity_j = scarcity_i.T
    scarcity_bonus = np.sqrt(scarcity_i * scarcity_j)

    score = (
        local_affinity ** 1.35
        * rank_affinity
        * regret_penalty
        * (0.82 + 0.18 * global_affinity)
        * (
            1.0
            + 0.22 * two_sided_advantage
            + 0.16 * scarcity_bonus
            + 0.18 * mutual_candidate
        )
    )

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
