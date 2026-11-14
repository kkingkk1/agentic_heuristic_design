import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized, diversity-preserving TSP heuristic matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    raw = np.where(finite, distances, np.inf)
    reverse = raw.T

    forward_valid = np.isfinite(raw)
    backward_valid = np.isfinite(reverse)
    both_valid = forward_valid & backward_valid

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both_valid] = 0.5 * (raw[both_valid] + reverse[both_valid])

    only_forward = forward_valid & ~backward_valid
    only_backward = forward_valid.T & ~backward_valid.T
    distance[only_forward] = raw[only_forward]
    distance[only_backward] = reverse[only_backward]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return scores

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        positive_fallback = finite_values[finite_values > 0.0]
        global_scale = (
            float(np.median(positive_fallback))
            if positive_fallback.size
            else 1.0
        )

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)

    valid_counts = valid.sum(axis=1)
    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.5 * np.sqrt(n)))),
    )

    local_window = sorted_distance[:, :neighborhood_size]
    local_window_valid = sorted_valid[:, :neighborhood_size]

    local_values = np.where(
        local_window_valid,
        local_window,
        global_scale,
    )

    local_scale = np.median(local_values, axis=1)

    positive_local = local_values > 0.0
    positive_counts = positive_local.sum(axis=1)

    positive_medians = np.divide(
        np.sum(np.where(positive_local, local_values, 0.0), axis=1),
        np.maximum(positive_counts, 1),
    )

    local_scale = np.where(
        (positive_counts > 0) & np.isfinite(positive_medians),
        positive_medians,
        local_scale,
    )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale >= 0.0),
        local_scale,
        global_scale,
    )

    local_floor = max(0.01 * global_scale, eps)
    local_scale = np.maximum(local_scale, local_floor)

    positive_all = finite_values[finite_values > 0.0]
    if positive_all.size:
        q25, q75 = np.percentile(positive_all, [25.0, 75.0])
    else:
        q25, q75 = 0.0, global_scale

    spread = max(float(q75 - q25), 0.1 * global_scale, eps)

    relative_local = np.divide(
        distance,
        np.sqrt(local_scale[:, None] * local_scale[None, :]),
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    relative_global = np.divide(
        distance,
        global_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=valid,
    )

    relative_local = np.clip(relative_local, 0.0, 40.0)
    relative_global = np.clip(relative_global, 0.0, 40.0)

    shortness = np.exp(-0.62 * relative_local)
    global_potential = np.exp(-0.28 * relative_global)

    ranks_sorted = np.arange(1, n + 1, dtype=float)[None, :]
    rank_values = np.where(
        sorted_valid,
        1.0 / ranks_sorted,
        0.0,
    )

    ranks = np.zeros((n, n), dtype=float)
    np.put_along_axis(ranks, order, rank_values, axis=1)

    rank_agreement = np.sqrt(ranks * ranks.T)

    first_distance = np.where(
        sorted_valid[:, 0],
        sorted_distance[:, 0],
        global_scale,
    )

    second_distance = np.where(
        sorted_valid[:, 1],
        sorted_distance[:, 1],
        first_distance,
    ) if n > 1 else first_distance

    third_distance = np.where(
        sorted_valid[:, 2],
        sorted_distance[:, 2],
        second_distance,
    ) if n > 2 else second_distance

    first_gap = np.divide(
        second_distance - first_distance,
        np.maximum(local_scale, eps),
    )
    second_gap = np.divide(
        third_distance - second_distance,
        np.maximum(local_scale, eps),
    )

    first_gap = np.clip(first_gap, 0.0, 4.0)
    second_gap = np.clip(second_gap, 0.0, 4.0)

    endpoint_regret = np.sqrt(
        (0.35 + first_gap[:, None])
        * (0.35 + first_gap[None, :])
    )
    endpoint_regret = endpoint_regret / (
        1.0 + 0.5 * relative_local
    )
    endpoint_regret = np.clip(endpoint_regret / 2.0, 0.0, 1.0)

    branching = np.minimum(valid_counts, neighborhood_size).astype(float)
    branching = branching / max(float(neighborhood_size), 1.0)

    scarcity = 1.0 - branching
    scarcity_pair = np.sqrt(
        np.maximum(scarcity[:, None], 0.0)
        * np.maximum(scarcity[None, :], 0.0)
    )

    edge_z = np.divide(
        distance - q25,
        spread,
        out=np.zeros_like(distance),
        where=valid,
    )
    bridge_potential = 1.0 / (
        1.0 + np.exp(np.clip(edge_z, -20.0, 20.0))
    )

    candidate_mask = (
        valid
        & (ranks > 0.0)
        & (
            (ranks <= 1.0 / 3.0)
            | (distance <= q75 + spread)
        )
    )

    candidate_strength = np.where(
        candidate_mask,
        np.exp(-0.38 * relative_local),
        0.0,
    )

    continuation = candidate_strength @ candidate_strength
    continuation = np.divide(
        continuation,
        np.maximum(
            np.sqrt(
                np.maximum(
                    candidate_strength.sum(axis=1)[:, None]
                    * candidate_strength.sum(axis=1)[None, :],
                    eps,
                )
            ),
            eps,
        ),
    )
    continuation = np.clip(continuation, 0.0, 1.0)

    two_step_potential = np.sqrt(
        np.maximum(continuation, 0.0)
        * np.maximum(continuation.T, 0.0)
    )

    row_weights = np.exp(
        -np.minimum(relative_local, 30.0)
    )
    row_weights = np.where(valid, row_weights, 0.0)

    row_sums = row_weights.sum(axis=1, keepdims=True)
    preferences = np.divide(
        row_weights,
        np.maximum(row_sums, eps),
        out=np.zeros_like(row_weights),
        where=row_sums > 0.0,
    )

    uniform = np.divide(
        valid.astype(float),
        np.maximum(valid_counts[:, None], 1.0),
    )

    adaptive_exploration = 0.06 + 0.08 * scarcity_pair

    mutual_preference = np.sqrt(
        np.maximum(preferences, 0.0)
        * np.maximum(preferences.T, 0.0)
    )

    scores = (
        0.34 * shortness
        + 0.16 * mutual_preference
        + 0.12 * rank_agreement
        + 0.13 * endpoint_regret
        + 0.12 * two_step_potential
        + 0.08 * global_potential
        + 0.05 * bridge_potential
    )

    scores += adaptive_exploration * (
        0.55 * uniform
        + 0.25 * global_potential
        + 0.20 * bridge_potential
    )

    scores += 0.035 * scarcity_pair * endpoint_regret

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
