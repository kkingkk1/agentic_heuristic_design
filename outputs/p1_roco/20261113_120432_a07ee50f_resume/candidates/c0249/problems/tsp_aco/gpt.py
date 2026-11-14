import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    valid_input = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    safe = np.where(valid_input, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    values = safe[valid]
    positive_values = values[values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(values)), 1.0)

    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)[None, :]

    column_order = np.argsort(safe, axis=0, kind="stable")
    column_rank = np.full((n, n), float(n), dtype=float)
    column_rank[
        column_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    usable_k = min(3, n - 1)
    nearest_idx = order[:, :usable_k]
    nearest_distances = np.take_along_axis(safe, nearest_idx, axis=1)
    nearest_valid = np.isfinite(nearest_distances)
    counts = nearest_valid.sum(axis=1)

    local_scale = np.sum(
        np.where(nearest_valid, nearest_distances, 0.0),
        axis=1,
    ) / np.maximum(counts, 1)

    local_scale = np.where(
        counts > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    pair_scale = np.maximum(pair_scale, eps)

    normalized_distance = safe / pair_scale
    distance_log = -np.clip(normalized_distance, 0.0, 50.0)

    rank_signal = -0.5 * (
        np.log1p(row_rank) + np.log1p(column_rank)
    )

    endpoint_affinity = -0.5 * (
        np.clip(safe / local_scale[:, None], 0.0, 50.0)
        + np.clip(safe / local_scale[None, :], 0.0, 50.0)
    )

    alternative_scale = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    )
    regret = np.maximum(
        safe / np.maximum(alternative_scale, eps) - 1.0,
        0.0,
    )
    regret_penalty = -0.14 * np.clip(regret, 0.0, 4.0)

    top_k = min(3, n - 1)
    neighborhood = np.zeros((n, n), dtype=bool)
    neighborhood[
        np.arange(n)[:, None],
        order[:, :top_k],
    ] = True
    neighborhood &= valid

    degree = np.maximum(
        neighborhood.sum(axis=1).astype(float),
        1.0,
    )
    overlap = neighborhood.astype(float) @ neighborhood.astype(float).T
    overlap /= np.sqrt(degree[:, None] * degree[None, :])
    overlap = np.clip(overlap, 0.0, 1.0)

    mutual_top = (
        valid
        & neighborhood
        & neighborhood.T
    ).astype(float)

    nearest_row = order[:, 0]
    nearest_column = column_order[0, :]
    mutual_nearest = (
        valid
        & (nearest_row[:, None] == np.arange(n)[None, :])
        & (nearest_column[None, :] == np.arange(n)[:, None])
    ).astype(float)

    # Small bonuses preserve useful exploration without dominating distance signals.
    mutual_signal = (
        0.10 * mutual_top
        + 0.14 * mutual_nearest
        + 0.05 * np.sqrt(mutual_top * mutual_top.T)
    )

    # Penalize redundant local concentration mildly and reward weak bridge evidence.
    diversity_signal = (
        -0.045 * overlap
        + 0.035 * (1.0 - overlap) * (0.35 + 0.65 * mutual_top)
    )

    # Bounded endpoint-regret support retains Explorer's long-term necessity cue.
    nearest = np.full(n, np.inf, dtype=float)
    second_nearest = np.full(n, np.inf, dtype=float)

    for i in range(n):
        row_values = safe[i, valid[i]]
        if row_values.size:
            ordered_values = np.sort(row_values)
            nearest[i] = ordered_values[0]
            second_nearest[i] = ordered_values[
                min(1, ordered_values.size - 1)
            ]

    endpoint_regret = np.maximum(second_nearest - nearest, 0.0)
    endpoint_regret /= np.maximum(local_scale, eps)
    endpoint_regret = np.clip(endpoint_regret, 0.0, 4.0)

    long_term_signal = 0.035 * np.sqrt(
        endpoint_regret[:, None] * endpoint_regret[None, :]
    )

    log_scores = (
        0.46 * distance_log
        + 0.23 * rank_signal
        + 0.23 * endpoint_affinity
        + regret_penalty
        + mutual_signal
        + diversity_signal
        + long_term_signal
    )

    log_scores[~valid] = -np.inf

    finite_valid_scores = log_scores[valid]
    score_max = float(np.max(finite_valid_scores))

    scores = np.zeros((n, n), dtype=float)
    scores[valid] = np.exp(
        np.clip(log_scores[valid] - score_max, -50.0, 0.0)
    )

    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores[~valid] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
