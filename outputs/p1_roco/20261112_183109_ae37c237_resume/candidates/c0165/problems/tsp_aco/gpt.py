import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge-priority indicators for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    eps = np.finfo(float).eps

    valid_directed = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid_directed, False)

    # Use only edges available in both directions.
    valid = valid_directed & valid_directed.T
    if not np.any(valid):
        return scores

    d = 0.5 * (distances + distances.T)
    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    positive_values = d[valid & (d > 0.0)]
    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        finite_values = d[valid]
        global_scale = (
            float(np.median(finite_values))
            if finite_values.size
            else 1.0
        )

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_values = d[valid]
    finite_max = float(np.max(finite_values))
    fallback = max(finite_max, global_scale) * 4.0 + global_scale

    working = np.where(valid, d, fallback)
    np.fill_diagonal(working, fallback)

    order = np.argsort(working, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    ranks[~valid] = np.inf

    neighbor_count = valid.sum(axis=1)
    k = min(7, n - 1)

    local_scale = np.full(n, global_scale, dtype=float)
    nearest = np.full(n, np.inf, dtype=float)
    second = np.full(n, np.inf, dtype=float)

    for i in range(n):
        row_values = d[i, valid[i]]
        if row_values.size == 0:
            continue

        row_values = np.sort(row_values)
        nearest[i] = row_values[0]
        second[i] = row_values[1] if row_values.size > 1 else row_values[0]

        take = min(k, row_values.size)
        local_scale[i] = float(np.median(row_values[:take]))

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    safe_nearest = np.where(
        np.isfinite(nearest),
        np.maximum(nearest, 0.0),
        global_scale,
    )
    safe_second = np.where(
        np.isfinite(second),
        np.maximum(second, safe_nearest),
        safe_nearest,
    )

    regret = np.divide(
        safe_second - safe_nearest,
        np.maximum(safe_second, eps),
        out=np.zeros(n, dtype=float),
        where=safe_second > 0.0,
    )
    regret = np.clip(regret, 0.0, 1.0)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        d,
        pair_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid & (pair_scale > 0.0),
    )

    distance_term = np.exp(
        -0.9 * np.clip(relative_distance, 0.0, 60.0)
    )

    rank_strength = np.exp(
        -ranks / max(1.0, 0.9 * np.sqrt(float(n)))
    )
    bidirectional_rank = np.sqrt(
        rank_strength * rank_strength.T
    )
    bidirectional_rank[~valid] = 0.0

    endpoint_quality = np.exp(
        -0.5
        * np.clip(
            d / local_scale[:, None]
            + d / local_scale[None, :],
            0.0,
            60.0,
        )
    )

    regret_bonus = 1.0 + 0.42 * (
        regret[:, None] + regret[None, :]
    )

    nearest_neighbor = np.argmin(working, axis=1)
    mutual_nearest = (
        nearest_neighbor[:, None] == np.arange(n)[None, :]
    ) & (
        nearest_neighbor[None, :] == np.arange(n)[:, None]
    )
    mutual_bonus = np.where(mutual_nearest, 1.50, 1.0)

    accessibility = np.exp(
        -np.clip(d / local_scale[:, None], 0.0, 60.0)
    )
    accessibility[~valid] = 0.0

    ambiguity = accessibility.sum(axis=1)
    scarcity = 1.0 / np.sqrt(
        1.0 + ambiguity[:, None] + ambiguity[None, :]
    )
    scarcity_bonus = 1.0 + 0.30 * scarcity

    long_edge_penalty = np.exp(
        -0.38 * np.maximum(relative_distance - 1.0, 0.0)
    )

    scores = (
        distance_term
        * (0.48 + 0.30 * bidirectional_rank + 0.22 * endpoint_quality)
        * regret_bonus
        * mutual_bonus
        * scarcity_bonus
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
