import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    eps = np.finfo(float).eps
    diagonal = np.eye(n, dtype=bool)

    directed_valid = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    forward = np.where(directed_valid, distances, np.inf)
    backward = np.where(directed_valid.T, distances.T, np.inf)

    both_valid = np.isfinite(forward) & np.isfinite(backward)
    only_forward = np.isfinite(forward) & ~np.isfinite(backward)
    only_backward = np.isfinite(backward) & ~np.isfinite(forward)

    safe = np.full((n, n), np.inf, dtype=float)
    safe[both_valid] = 0.5 * (
        forward[both_valid] + backward[both_valid]
    )
    safe[only_forward] = forward[only_forward]
    safe[only_backward] = backward[only_backward]

    np.fill_diagonal(safe, np.inf)
    valid = np.isfinite(safe)

    if not np.any(valid):
        return scores

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.full((n, n), float(n), dtype=float)
    top_k = min(max(3, int(np.ceil(1.35 * np.sqrt(n))),), n - 1)

    top_mask = np.zeros((n, n), dtype=bool)
    nearest_mask = np.zeros((n, n), dtype=bool)

    local_scale = np.full(n, global_scale, dtype=float)
    second_distance = np.full(n, global_scale, dtype=float)
    candidate_counts = valid.sum(axis=1).astype(float)

    for i in range(n):
        candidates = order[i][valid[i, order[i]]]

        if candidates.size == 0:
            continue

        row_values = safe[i, candidates]
        ranks[i, candidates] = 1.0 / (
            1.0 + np.arange(candidates.size, dtype=float)
        )

        top_candidates = candidates[:min(top_k, candidates.size)]
        top_mask[i, top_candidates] = True
        nearest_mask[i, candidates[0]] = True

        if candidates.size > 1:
            second_distance[i] = row_values[1]
        else:
            second_distance[i] = row_values[0]

        positive_neighbors = row_values[row_values > 0.0]
        if positive_neighbors.size:
            local_scale[i] = float(
                np.median(
                    positive_neighbors[
                        :min(6, positive_neighbors.size)
                    ]
                )
            )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, 0.05 * global_scale)
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    relative_distance = np.divide(
        safe,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    endpoint_ratio = np.divide(
        safe,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_ratio = np.clip(endpoint_ratio, 0.0, 50.0)

    reverse_endpoint_ratio = np.divide(
        safe,
        np.maximum(local_scale[None, :], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    reverse_endpoint_ratio = np.clip(reverse_endpoint_ratio, 0.0, 50.0)

    distance_score = np.exp(
        -np.clip(relative_distance ** 1.18, 0.0, 50.0)
    )

    endpoint_affinity = np.sqrt(
        np.exp(-endpoint_ratio)
        * np.exp(-reverse_endpoint_ratio)
    )

    row_preferences = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        logits = (
            -1.05 * relative_distance[i, row_valid]
            -0.12 * endpoint_ratio[i, row_valid]
        )

        maximum = float(np.max(logits))
        weights = np.exp(logits - maximum)
        total = float(np.sum(weights))

        if np.isfinite(total) and total > 0.0:
            row_preferences[i, row_valid] = weights / total

    mutual_preference = np.sqrt(
        row_preferences * row_preferences.T
    )

    rank_agreement = np.sqrt(
        np.clip(ranks * ranks.T, 0.0, 1.0)
    )

    mutual_top = (
        top_mask
        & top_mask.T
        & valid
    ).astype(float)

    mutual_nearest = (
        nearest_mask
        & nearest_mask.T
        & valid
    ).astype(float)

    shared_neighbors = (
        top_mask.astype(float) @ top_mask.astype(float).T
    )
    shared_neighbor_signal = np.clip(
        shared_neighbors / max(float(top_k), 1.0),
        0.0,
        1.0,
    )

    second_pair = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    )

    scarcity_ratio = np.divide(
        second_pair,
        safe + 0.5 * global_scale,
        out=np.zeros((n, n), dtype=float),
        where=valid,
    )
    scarcity_bonus = 1.0 + 0.16 * np.clip(
        scarcity_ratio - 0.75,
        0.0,
        3.0,
    )

    opportunity_ratio = np.divide(
        safe,
        np.maximum(
            np.minimum(
                local_scale[:, None],
                local_scale[None, :],
            ),
            eps,
        ),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    opportunity_penalty = 1.0 / (
        1.0
        + 0.08 * np.clip(opportunity_ratio - 1.0, 0.0, 6.0)
    )

    density_factor = np.sqrt(
        np.clip(
            candidate_counts[:, None]
            * candidate_counts[None, :]
            / max(float(n - 1) ** 2, 1.0),
            0.35,
            1.0,
        )
    )

    scores = (
        distance_score
        * endpoint_affinity
        * mutual_preference
        * (0.52 + 0.48 * rank_agreement)
        * (0.78 + 0.22 * np.maximum(
            row_preferences,
            row_preferences.T,
        ))
        * (1.0 + 0.58 * mutual_top)
        * (1.0 + 0.34 * mutual_nearest)
        * (1.0 + 0.08 * shared_neighbor_signal)
        * scarcity_bonus
        * opportunity_penalty
        * (0.90 + 0.10 * density_factor)
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
