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
    raw_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(raw_valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(raw_valid, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    row_order = np.argsort(safe, axis=1, kind="stable")
    row_rank = np.full((n, n), float(n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    valid_rank = np.where(valid, row_rank, np.inf)

    usable = min(5, n - 1)
    neighborhood = np.take_along_axis(
        safe,
        row_order[:, :usable],
        axis=1,
    )
    neighborhood_valid = np.isfinite(neighborhood)

    local_scale = np.sum(
        np.where(neighborhood_valid, neighborhood, 0.0),
        axis=1,
    ) / np.maximum(np.sum(neighborhood_valid, axis=1), 1)

    local_scale = np.where(
        np.sum(neighborhood_valid, axis=1) > 0,
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    normalized_distance = safe / np.maximum(pair_scale, eps)
    local_distance_score = np.exp(
        -np.clip(normalized_distance, 0.0, 50.0)
    )

    global_distance_score = np.exp(
        -np.clip(safe / global_scale, 0.0, 50.0)
    )

    row_rank_score = np.exp(
        -np.clip(valid_rank / max(float(min(5, n - 1)), 1.0), 0.0, 50.0)
    )
    reciprocal_rank = np.sqrt(
        row_rank_score * row_rank_score.T
    )

    endpoint_relative_score = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 50.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 50.0))
    )

    candidate_masks = []
    for k in (3, 4, 5):
        kk = min(k, n - 1)
        mask = np.zeros((n, n), dtype=bool)
        mask[
            np.arange(n)[:, None],
            row_order[:, :kk],
        ] = True
        mask &= valid
        candidate_masks.append(mask)

    candidate_support = np.mean(
        np.stack(candidate_masks, axis=0),
        axis=0,
    )

    local_overlap = np.zeros((n, n), dtype=float)
    for mask in candidate_masks:
        overlap = mask.astype(float) @ mask.astype(float).T
        degree = np.maximum(mask.sum(axis=1).astype(float), 1.0)
        normalized_overlap = overlap / np.sqrt(
            degree[:, None] * degree[None, :]
        )
        local_overlap += normalized_overlap

    local_overlap /= max(len(candidate_masks), 1)

    diversity_factor = 1.0 - 0.18 * np.clip(
        local_overlap * candidate_support,
        0.0,
        1.0,
    )

    sorted_valid_distances = np.where(valid, safe, np.inf)
    nearest = np.min(sorted_valid_distances, axis=1)

    second_nearest = np.full(n, np.inf, dtype=float)
    for i in range(n):
        values = sorted_valid_distances[i]
        finite = values[np.isfinite(values)]
        if finite.size >= 2:
            second_nearest[i] = finite[1]
        elif finite.size == 1:
            second_nearest[i] = finite[0]

    endpoint_regret = np.maximum(
        second_nearest - nearest,
        0.0,
    ) / np.maximum(local_scale, eps)

    regret_signal = np.sqrt(
        (1.0 + np.clip(endpoint_regret[:, None], 0.0, 4.0))
        * (1.0 + np.clip(endpoint_regret[None, :], 0.0, 4.0))
    )

    long_term_factor = 1.0 + 0.10 * np.clip(
        regret_signal - 1.0,
        0.0,
        2.0,
    ) * candidate_support

    global_local_mix = (
        0.62 * local_distance_score
        + 0.23 * global_distance_score
        + 0.15 * endpoint_relative_score
    )

    rank_factor = 0.55 + 0.45 * reciprocal_rank

    scores = (
        global_local_mix
        * rank_factor
        * (0.90 + 0.10 * candidate_support)
        * diversity_factor
        * long_term_factor
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
