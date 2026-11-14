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
    finite = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    safe = np.where(finite, distances, np.inf)
    safe = np.minimum(safe, safe.T)
    valid = np.isfinite(safe) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = safe[valid & (safe > 0.0)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = safe[valid]
        global_scale = max(float(np.median(finite_values)), 1.0)

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    masked = np.where(valid, safe, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(masked, order, axis=1)

    usable = min(5, n - 1)
    nearest = sorted_distances[:, :usable]
    nearest_valid = np.isfinite(nearest)

    local_scale = np.divide(
        np.where(nearest_valid, nearest, 0.0).sum(axis=1),
        np.maximum(nearest_valid.sum(axis=1), 1),
    )
    local_scale = np.where(
        nearest_valid.any(axis=1),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(local_scale[:, None] * local_scale[None, :])

    normalized_distance = safe / np.maximum(pair_scale, eps)
    distance_score = np.exp(-np.clip(normalized_distance, 0.0, 40.0))

    endpoint_score = np.sqrt(
        np.exp(-np.clip(safe / local_scale[:, None], 0.0, 40.0))
        * np.exp(-np.clip(safe / local_scale[None, :], 0.0, 40.0))
    )

    candidate_k = min(max(3, int(np.ceil(np.sqrt(n))), 6), n - 1)
    candidates = np.zeros((n, n), dtype=bool)
    candidates[
        np.arange(n)[:, None],
        order[:, :candidate_k],
    ] = True
    candidates &= valid

    reverse_candidates = candidates.T
    mutual_support = (candidates & reverse_candidates).astype(float)

    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        order,
    ] = np.arange(n, dtype=float)

    rank_score = 1.0 / np.sqrt(
        (1.0 + row_rank) * (1.0 + row_rank.T)
    )
    rank_score = 0.45 + 0.55 * rank_score

    best_distance = np.min(masked, axis=1)
    second_distance = np.full(n, np.inf, dtype=float)

    if n > 2:
        second_distance = sorted_distances[:, 1]

    best_distance = np.where(np.isfinite(best_distance), best_distance, global_scale)
    second_distance = np.where(
        np.isfinite(second_distance),
        second_distance,
        best_distance + global_scale,
    )

    regret = np.clip(
        (second_distance - best_distance)
        / np.maximum(second_distance, global_scale),
        0.0,
        1.0,
    )

    regret_factor = 1.0 + 0.18 * np.sqrt(
        regret[:, None] * regret[None, :]
    )

    row_candidate_count = candidates.sum(axis=1).astype(float)
    candidate_popularity = candidates.sum(axis=0).astype(float)

    popularity_reference = max(
        float(np.mean(candidate_popularity)),
        1.0,
    )
    popularity_ratio = np.clip(
        candidate_popularity / popularity_reference,
        0.0,
        4.0,
    )

    popularity_factor = 1.0 / np.sqrt(
        (1.0 + 0.10 * popularity_ratio[:, None])
        * (1.0 + 0.10 * popularity_ratio[None, :])
    )

    shared = candidates.astype(float) @ candidates.astype(float).T
    union = (
        row_candidate_count[:, None]
        + row_candidate_count[None, :]
        - shared
    )
    overlap = shared / np.maximum(union, 1.0)

    complement_factor = 1.0 + 0.12 * (
        1.0 - np.clip(overlap, 0.0, 1.0)
    ) * mutual_support

    relative_cost = safe / np.maximum(
        np.minimum(
            local_scale[:, None],
            local_scale[None, :],
        ),
        eps,
    )
    opportunity_factor = 1.0 / (
        1.0 + 0.10 * np.clip(relative_cost - 1.0, 0.0, 5.0)
    )

    isolation_factor = 1.0 + 0.12 * np.clip(
        pair_scale / np.maximum(safe + global_scale, eps),
        0.0,
        1.0,
    )

    scores = (
        distance_score
        * endpoint_score
        * rank_score
        * (1.0 + 0.16 * mutual_support)
        * regret_factor
        * popularity_factor
        * complement_factor
        * opportunity_factor
        * isolation_factor
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
