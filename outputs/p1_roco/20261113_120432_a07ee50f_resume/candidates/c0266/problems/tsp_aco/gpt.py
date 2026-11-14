import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized prior for stochastic TSP edge sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & (distances >= 0.0) & ~diagonal

    if not np.any(finite):
        return result

    forward = np.where(finite, distances, np.inf)
    backward = forward.T

    valid_forward = np.isfinite(forward)
    valid_backward = np.isfinite(backward)

    both = valid_forward & valid_backward
    only_forward = valid_forward & ~valid_backward
    only_backward = valid_backward & ~valid_forward

    edge_distance = np.full((n, n), np.inf, dtype=float)
    edge_distance[both] = 0.5 * (
        forward[both] + backward[both]
    )
    edge_distance[only_forward] = forward[only_forward]
    edge_distance[only_backward] = backward[only_backward]

    np.fill_diagonal(edge_distance, np.inf)
    valid = np.isfinite(edge_distance)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    finite_values = edge_distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    local_temperature = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = edge_distance[i, valid[i]]
        if values.size == 0:
            continue

        positive = values[values > 0.0]
        if positive.size:
            center = float(np.median(positive))
            spread = float(np.median(np.abs(positive - center)))
            local_temperature[i] = max(
                0.5 * center + spread,
                global_scale * 0.08,
                eps,
            )
        else:
            local_temperature[i] = max(
                float(np.median(values)),
                global_scale * 0.08,
                eps,
            )

    local_temperature = np.maximum(
        np.where(
            np.isfinite(local_temperature),
            local_temperature,
            global_scale,
        ),
        global_scale * 0.05,
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logits = -edge_distance / local_temperature[:, None]
        logits[~valid] = -np.inf

        row_max = np.max(logits, axis=1, keepdims=True)
        shifted = logits - row_max
        weights = np.exp(np.clip(shifted, -60.0, 0.0))
        weights[~valid] = 0.0

        row_totals = np.sum(weights, axis=1, keepdims=True)
        probabilities = weights / np.maximum(row_totals, eps)

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    reciprocal_support = np.sqrt(
        probabilities * probabilities.T
    )

    endpoint_activity = np.sum(
        np.sqrt(probabilities * probabilities.T),
        axis=1,
    )
    endpoint_activity = endpoint_activity / np.maximum(
        np.max(endpoint_activity),
        eps,
    )

    balance = np.sqrt(
        np.maximum(endpoint_activity[:, None], eps)
        * np.maximum(endpoint_activity[None, :], eps)
    )
    balance = 1.0 / np.sqrt(0.55 + balance)

    safe_distance = np.where(valid, edge_distance, np.inf)
    expanded_left = safe_distance[:, :, None]
    expanded_right = safe_distance[None, :, :]

    two_hop = expanded_left + expanded_right
    two_hop[:, np.arange(n), np.arange(n)] = np.inf

    intermediary_valid = (
        np.isfinite(expanded_left)
        & np.isfinite(expanded_right)
    )

    two_hop = np.where(intermediary_valid, two_hop, np.inf)
    best_two_hop = np.min(two_hop, axis=2)

    replacement_gap = (
        best_two_hop - safe_distance
    ) / np.maximum(global_scale, eps)

    replacement_gap = np.nan_to_num(
        replacement_gap,
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    )

    bridge_value = 1.0 / (
        1.0 + np.exp(-np.clip(replacement_gap, -8.0, 8.0))
    )

    no_replacement = ~np.isfinite(best_two_hop)
    bridge_value[no_replacement & valid] = 1.0
    bridge_value[~valid] = 0.0

    direct_cost = np.exp(
        -np.clip(
            safe_distance / np.maximum(global_scale, eps),
            0.0,
            50.0,
        )
    )

    scores = (
        np.power(np.maximum(reciprocal_support, 0.0), 0.72)
        * (0.35 + 0.65 * bridge_value)
        * (0.72 + 0.28 * direct_cost)
        * balance
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
