import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-adaptive edge-priority indicators for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    finite_positive = np.isfinite(distances) & (distances > 0)
    valid = finite_positive & finite_positive.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    eps = np.finfo(float).eps
    ordered = np.sort(d, axis=1)

    finite_values = d[np.isfinite(d)]
    if finite_values.size == 0:
        return result

    global_scale = float(np.median(finite_values))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    k = min(6, n - 1)
    neighborhood = ordered[:, :k]
    finite_neighbors = np.isfinite(neighborhood)

    neighbor_count = finite_neighbors.sum(axis=1)
    neighbor_values = np.where(finite_neighbors, neighborhood, 0.0)
    neighbor_median = np.median(
        np.where(finite_neighbors, neighborhood, np.nan),
        axis=1,
    )

    neighbor_mean = np.divide(
        neighbor_values.sum(axis=1),
        np.maximum(neighbor_count, 1),
    )

    local_scale = np.where(
        np.isfinite(neighbor_median) & (neighbor_median > 0),
        neighbor_median,
        neighbor_mean,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    geometric_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=np.isfinite(d),
    )

    clipped_relative = np.clip(relative_distance, 0.0, 60.0)

    distance_attractiveness = np.exp(
        -0.95 * np.power(clipped_relative, 1.08)
    )

    row_logits = np.divide(
        -d,
        local_scale[:, None],
        out=np.full((n, n), -np.inf, dtype=float),
        where=np.isfinite(d),
    )
    row_logits = np.clip(row_logits, -60.0, 0.0)

    row_weights = np.exp(row_logits)
    row_weights[~valid] = 0.0

    row_totals = row_weights.sum(axis=1, keepdims=True)
    row_probability = np.divide(
        row_weights,
        row_totals,
        out=np.zeros_like(row_weights),
        where=row_totals > 0,
    )

    mutual_choice = np.sqrt(
        row_probability * row_probability.T
    )

    row_order = np.argsort(d, axis=1, kind="stable")
    ranks = np.full((n, n), np.inf, dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], row_order] = rank_values[None, :]
    ranks[~valid] = np.inf

    rank_scale = max(1.0, 0.95 * np.sqrt(float(n)))
    rank_support = np.exp(
        -0.5 * (ranks + ranks.T) / rank_scale
    )
    rank_support[~valid] = 0.0

    nearest = row_order[:, 0]
    nearest_distance = ordered[:, 0]
    second_distance = (
        ordered[:, 1] if n > 2 else nearest_distance
    )

    nearest_safe = np.where(
        np.isfinite(nearest_distance),
        nearest_distance,
        global_scale,
    )
    second_safe = np.where(
        np.isfinite(second_distance),
        second_distance,
        nearest_safe,
    )

    regret = np.divide(
        second_safe - nearest_safe,
        np.maximum(second_safe, eps),
        out=np.zeros(n, dtype=float),
        where=np.isfinite(second_safe),
    )
    regret = np.clip(np.nan_to_num(regret), 0.0, 1.0)

    mutual_nearest = np.zeros((n, n), dtype=float)
    for i in range(n):
        if not np.isfinite(nearest_distance[i]):
            continue
        j = int(nearest[i])
        if (
            0 <= j < n
            and valid[i, j]
            and np.isfinite(nearest_distance[j])
            and int(nearest[j]) == i
        ):
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    long_edge_penalty = np.exp(
        -0.24 * np.maximum(relative_distance - 1.0, 0.0)
    )

    regret_bonus = 1.0 + 0.28 * (
        regret[:, None] + regret[None, :]
    )

    score = (
        distance_attractiveness
        * (0.72 + 0.28 * mutual_choice)
        * (0.78 + 0.22 * rank_support)
        * (1.0 + 0.22 * mutual_nearest)
        * regret_bonus
        * long_edge_penalty
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(float, copy=False)
