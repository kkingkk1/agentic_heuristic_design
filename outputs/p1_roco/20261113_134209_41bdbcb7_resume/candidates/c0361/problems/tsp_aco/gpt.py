import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, normalized and diversity-preserving TSP heuristic matrix."""
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
    distance[both_valid] = 0.5 * (
        raw[both_valid] + reverse[both_valid]
    )

    only_forward = forward_valid & ~backward_valid
    only_backward = backward_valid & ~forward_valid
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
        global_scale = max(float(np.median(finite_values)), 1.0)

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)

    neighborhood_size = min(
        n - 1,
        max(3, int(np.ceil(1.25 * np.sqrt(n)))),
    )
    local_width = min(6, n - 1)

    local_window = sorted_distance[:, :local_width]
    local_window_valid = sorted_valid[:, :local_width]

    local_values = np.where(
        local_window_valid,
        local_window,
        global_scale,
    )
    local_scale = np.median(local_values, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, 0.05 * global_scale)

    positions = np.arange(n, dtype=float)[None, :]
    rank_sorted = np.where(
        sorted_valid,
        1.0 / (1.0 + positions),
        0.0,
    )
    ranks = np.zeros((n, n), dtype=float)
    np.put_along_axis(ranks, order, rank_sorted, axis=1)

    first_distance = np.where(
        sorted_valid[:, 0],
        sorted_distance[:, 0],
        global_scale,
    )
    if n > 1:
        second_distance = np.where(
            sorted_valid[:, 1],
            sorted_distance[:, 1],
            first_distance,
        )
    else:
        second_distance = first_distance

    regret = np.sqrt(
        np.maximum(second_distance[:, None], global_scale)
        * np.maximum(second_distance[None, :], global_scale)
    )
    regret = regret / (
        distance + 0.5 * global_scale
    )
    regret = np.clip(regret, 0.0, 2.5) / 2.5

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    relative_distance = np.divide(
        distance,
        np.maximum(pair_scale, eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    relative_distance = np.clip(relative_distance, 0.0, 50.0)

    endpoint_ratio = np.divide(
        distance,
        np.maximum(local_scale[:, None], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )
    endpoint_ratio_reverse = np.divide(
        distance,
        np.maximum(local_scale[None, :], eps),
        out=np.full((n, n), 50.0, dtype=float),
        where=valid,
    )

    endpoint_ratio = np.clip(endpoint_ratio, 0.0, 50.0)
    endpoint_ratio_reverse = np.clip(endpoint_ratio_reverse, 0.0, 50.0)

    logits = (
        -0.78 * relative_distance
        -0.10 * endpoint_ratio
    )

    row_max = np.max(
        np.where(valid, logits, -np.inf),
        axis=1,
        keepdims=True,
    )

    safe_logits = np.where(
        valid,
        logits - row_max,
        -np.inf,
    )

    local_contrast = np.divide(
        second_distance - first_distance,
        np.maximum(local_scale, eps),
    )
    local_contrast = np.clip(local_contrast, 0.0, 3.0)

    temperature = 1.05 + 0.30 * local_contrast
    weights = np.exp(
        safe_logits / temperature[:, None]
    )
    weights = np.where(valid, weights, 0.0)

    row_totals = np.sum(weights, axis=1, keepdims=True)
    preferences = np.divide(
        weights,
        np.maximum(row_totals, eps),
        out=np.zeros_like(weights),
        where=row_totals > 0.0,
    )

    candidate_counts = valid.sum(axis=1, keepdims=True).astype(float)
    uniform = np.divide(
        valid.astype(float),
        np.maximum(candidate_counts, 1.0),
    )

    preferences = 0.82 * preferences + 0.18 * uniform

    mutual_preference = np.sqrt(
        preferences * preferences.T
    )

    rank_agreement = np.sqrt(
        ranks * ranks.T
    )

    shortness = np.exp(
        -0.74 * relative_distance
    )

    global_potential = np.exp(
        -0.42 * np.divide(
            distance,
            global_scale,
            out=np.full((n, n), 50.0, dtype=float),
            where=valid,
        )
    )

    endpoint_affinity = np.sqrt(
        np.exp(-0.42 * endpoint_ratio)
        * np.exp(-0.42 * endpoint_ratio_reverse)
    )

    scores = (
        0.43 * shortness
        + 0.19 * mutual_preference
        + 0.14 * rank_agreement
        + 0.10 * regret
        + 0.08 * global_potential
    )

    scores *= (
        0.78
        + 0.22 * endpoint_affinity
    )

    exploration_floor = 0.035 * (
        0.5 * global_potential
        + 0.5 * uniform
    )
    scores = scores + exploration_floor

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
