import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(directed_valid):
        return scores

    directed = np.where(directed_valid, distances, np.inf)
    reverse = directed.T

    valid = np.isfinite(directed) | np.isfinite(reverse)
    safe = np.minimum(directed, reverse)
    safe[~valid] = np.inf
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe)
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    top_k = min(max(3, int(np.ceil(np.sqrt(n)))), n - 1)
    top_neighbors = np.zeros((n, n), dtype=bool)

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]

        if candidates.size == 0:
            continue

        values = safe[i, candidates]
        ranks[i, candidates] = np.arange(
            candidates.size,
            dtype=float,
        )
        top_neighbors[i, candidates[:top_k]] = True

        second_distance[i] = (
            values[1] if values.size > 1 else values[0]
        )

        positive = values[values > 0.0]
        if positive.size:
            local_scale[i] = float(
                np.median(positive[: min(5, positive.size)])
            )
        else:
            local_scale[i] = global_scale

    local_scale = np.maximum(
        np.where(np.isfinite(local_scale), local_scale, global_scale),
        global_scale * 0.05,
    )

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    normalized_distance = safe / np.maximum(pair_scale, eps)
    normalized_distance = np.clip(normalized_distance, 0.0, 50.0)

    distance_kernel = np.exp(
        -1.12 * normalized_distance ** 1.12
    )

    endpoint_kernel = np.sqrt(
        np.exp(
            -np.clip(
                safe / np.maximum(local_scale[:, None], eps),
                0.0,
                50.0,
            )
        )
        * np.exp(
            -np.clip(
                safe / np.maximum(local_scale[None, :], eps),
                0.0,
                50.0,
            )
        )
    )

    rank_kernel = 1.0 / np.sqrt(
        (1.0 + ranks) * (1.0 + ranks.T)
    )

    mutual_top = (
        top_neighbors
        & top_neighbors.T
        & valid
    ).astype(float)

    nearest = np.argmin(safe, axis=1)
    mutual_nearest = np.zeros((n, n), dtype=float)

    for i, j in enumerate(nearest):
        if valid[i, j] and nearest[j] == i:
            mutual_nearest[i, j] = 1.0
            mutual_nearest[j, i] = 1.0

    valid_counts = valid.sum(axis=1).astype(float)
    scarcity = 1.0 / np.sqrt(np.maximum(valid_counts, 1.0))
    scarcity_pair = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )
    scarcity_pair /= max(float(np.max(scarcity_pair)), eps)

    opportunity_ratio = safe / np.maximum(
        np.minimum(
            local_scale[:, None],
            local_scale[None, :],
        ),
        eps,
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.10 * np.clip(opportunity_ratio - 1.0, 0.0, 5.0)
    )

    gap_ratio = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    ) / np.maximum(safe, eps)

    scarcity_bonus = 1.0 + 0.08 * np.clip(
        gap_ratio - 1.0,
        0.0,
        4.0,
    )

    scores = (
        distance_kernel
        * endpoint_kernel
        * (0.52 + 0.48 * rank_kernel)
        * (
            1.0
            + 0.30 * mutual_nearest
            + 0.14 * mutual_top
        )
        * opportunity_penalty
        * (0.90 + 0.20 * scarcity_pair)
        * scarcity_bonus
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
