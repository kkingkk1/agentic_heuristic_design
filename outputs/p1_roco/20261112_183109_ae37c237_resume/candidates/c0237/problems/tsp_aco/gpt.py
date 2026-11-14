import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-invariant edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    valid_values = symmetric_distance[valid]
    positive_values = valid_values[valid_values > 0.0]

    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif valid_values.size:
        global_scale = float(np.max(valid_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(valid_values))
    penalty = max(finite_max, global_scale) * 4.0 + global_scale

    work = np.where(valid, symmetric_distance, penalty)
    np.fill_diagonal(work, penalty)

    valid_count = np.sum(valid, axis=1)
    nearest_order = np.argmin(work, axis=1)
    nearest_distance = np.min(work, axis=1)

    nearest_two = np.partition(work, kth=1, axis=1)[:, :2]
    best_distance = nearest_two[:, 0]
    second_distance = nearest_two[:, 1]

    local_scale = np.where(
        valid_count >= 2,
        0.5 * (best_distance + second_distance),
        best_distance,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        symmetric_distance,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    endpoint_relative = 0.5 * (
        np.divide(
            symmetric_distance,
            local_scale[:, None],
            out=np.full((n, n), np.inf, dtype=float),
            where=local_scale[:, None] > 0.0,
        )
        + np.divide(
            symmetric_distance,
            local_scale[None, :],
            out=np.full((n, n), np.inf, dtype=float),
            where=local_scale[None, :] > 0.0,
        )
    )

    distance_term = np.exp(
        -np.clip(1.08 * relative_distance, 0.0, 60.0)
    )

    endpoint_preference = np.exp(
        -np.clip(0.92 * endpoint_relative, 0.0, 60.0)
    )

    endpoint_ratio_i = np.divide(
        symmetric_distance,
        local_scale[:, None],
        out=np.full((n, n), np.inf, dtype=float),
        where=local_scale[:, None] > 0.0,
    )
    endpoint_ratio_j = np.divide(
        symmetric_distance,
        local_scale[None, :],
        out=np.full((n, n), np.inf, dtype=float),
        where=local_scale[None, :] > 0.0,
    )

    rank_proxy = np.exp(
        -np.clip(
            0.90 * np.sqrt(
                np.maximum(endpoint_ratio_i * endpoint_ratio_j, 0.0)
            ),
            0.0,
            60.0,
        )
    )

    regret = np.divide(
        second_distance - best_distance,
        np.maximum(second_distance, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second_distance),
    )
    regret = np.clip(regret, 0.0, 1.0)

    regret_bonus = 1.0 + 0.36 * (
        regret[:, None] + regret[None, :]
    )

    indices = np.arange(n)
    mutual_nearest = (
        (nearest_order[:, None] == indices[None, :])
        & (nearest_order[None, :] == indices[:, None])
    )
    mutual_bonus = np.where(mutual_nearest, 1.38, 1.0)

    long_edge_penalty = np.exp(
        -0.48 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_term
        * (0.50 + 0.32 * endpoint_preference + 0.18 * rank_proxy)
        * regret_bonus
        * mutual_bonus
        * long_edge_penalty
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
