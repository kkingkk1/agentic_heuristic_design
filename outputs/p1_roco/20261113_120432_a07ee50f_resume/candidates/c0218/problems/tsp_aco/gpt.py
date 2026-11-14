import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    if not np.any(finite_nonnegative):
        return result

    eps = np.finfo(float).eps
    positive_values = raw[finite_nonnegative & (raw > 0.0)]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(raw[finite_nonnegative]))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    # Reconcile asymmetric finite entries into a safe undirected view.
    forward = np.where(finite_nonnegative, raw, np.inf)
    backward = forward.T
    distance = np.minimum(forward, backward)

    # If only one direction is available, retain that finite direction.
    distance = np.where(np.isfinite(forward), forward, distance)
    distance = np.where(np.isfinite(backward), np.minimum(distance, backward), distance)

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance) & ~diagonal

    if not np.any(valid):
        return result

    work = np.where(valid, distance, np.inf)

    neighborhood_size = min(5, n - 1)
    sorted_distance = np.sort(work, axis=1)
    local_values = sorted_distance[:, :neighborhood_size]

    local_scale = np.mean(
        np.where(np.isfinite(local_values), local_values, global_scale),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 0.05)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            eps,
        )
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized_distance = distance / pair_scale
        distance_score = np.exp(
            -np.clip(1.05 * normalized_distance, 0.0, 50.0)
        )

        endpoint_affinity = np.sqrt(
            np.exp(
                -np.clip(
                    distance / local_scale[:, None],
                    0.0,
                    50.0,
                )
            )
            * np.exp(
                -np.clip(
                    distance / local_scale[None, :],
                    0.0,
                    50.0,
                )
            )
        )

    # Reciprocal rank agreement.
    order = np.argsort(work, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    ranks[np.arange(n)[:, None], order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / np.power(1.0 + ranks, 0.90)
    reciprocal_rank = np.sqrt(rank_score * rank_score.T)

    # Mutual nearest-neighbor and mutual top-k reinforcement.
    nearest = np.argmin(work, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i in range(n):
        j = nearest[i]
        if np.isfinite(work[i, j]) and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    top_k = min(3, n - 1)
    top_mask = np.zeros((n, n), dtype=bool)
    top_mask[np.arange(n)[:, None], order[:, :top_k]] = True
    mutual_top = (top_mask & top_mask.T & valid).astype(float)

    # Bounded compensation for locally isolated endpoints.
    second_nearest = np.full(n, global_scale, dtype=float)
    for i in range(n):
        finite_row = sorted_distance[i][np.isfinite(sorted_distance[i])]
        if finite_row.size >= 2:
            second_nearest[i] = finite_row[1]
        elif finite_row.size == 1:
            second_nearest[i] = finite_row[0]

    gap = np.maximum(second_nearest, global_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        endpoint_scarcity = 1.0 + 0.24 * np.clip(
            gap[:, None] / np.maximum(distance, eps) - 1.0,
            0.0,
            4.0,
        )

    scarcity = np.sqrt(endpoint_scarcity * endpoint_scarcity.T)

    scores = (
        distance_score
        * endpoint_affinity
        * (0.50 + 0.50 * reciprocal_rank)
        * (1.0 + 0.38 * mutual_nearest + 0.14 * mutual_top)
        * scarcity
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
