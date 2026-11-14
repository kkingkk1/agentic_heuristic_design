import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return structural TSP edge priors based on one-hop/two-hop advantage."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    eps = np.finfo(float).eps
    finite = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(finite, False)

    reverse_finite = finite.T
    both = finite & reverse_finite
    forward_only = finite & ~reverse_finite
    backward_only = reverse_finite & ~finite

    distance = np.full((n, n), np.inf, dtype=float)
    distance[both] = 0.5 * (d[both] + d.T[both])
    distance[forward_only] = d[forward_only]
    distance[backward_only] = d.T[backward_only]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return result

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = max(float(np.median(finite_values)), 1.0)

    global_scale = max(global_scale, eps)

    degree = valid.sum(axis=1)
    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        neighbors = np.flatnonzero(valid[i])
        if neighbors.size:
            values = np.sort(distance[i, neighbors])
            local_scale[i] = float(
                np.median(values[: min(4, values.size)])
            )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    pair_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            eps,
        )
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_length = distance / pair_scale
        direct_quality = 1.0 / (
            1.0 + np.clip(relative_length, 0.0, 1e6)
        )

    if n > 2:
        via = distance[:, :, None] + distance[None, :, :]
        np.fill_diagonal(via[:, :, 0], np.inf)
        two_hop = np.min(via, axis=1)

        endpoint_index = np.arange(n)
        two_hop[endpoint_index, endpoint_index] = np.inf
    else:
        two_hop = np.full((n, n), np.inf, dtype=float)

    finite_two_hop = np.isfinite(two_hop)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        detour_ratio = np.divide(
            two_hop,
            np.maximum(distance, eps),
            out=np.zeros_like(distance),
            where=finite_two_hop & valid,
        )

    shortcut_advantage = np.clip(detour_ratio - 1.0, 0.0, 8.0)
    shortcut_score = shortcut_advantage / (1.0 + shortcut_advantage)

    with np.errstate(divide="ignore", invalid="ignore"):
        replacement_abundance = np.divide(
            distance,
            np.maximum(two_hop, eps),
            out=np.ones_like(distance),
            where=finite_two_hop & valid,
        )

    redundancy_penalty = 1.0 - 0.32 * np.clip(
        replacement_abundance,
        0.0,
        1.0,
    )

    endpoint_balance = np.sqrt(
        np.minimum(
            local_scale[:, None] / np.maximum(pair_scale, eps),
            local_scale[None, :] / np.maximum(pair_scale, eps),
        )
    )
    endpoint_balance = np.clip(endpoint_balance, 0.5, 1.0)

    degree_factor = np.sqrt(
        np.maximum(degree[:, None], 1.0)
        * np.maximum(degree[None, :], 1.0)
    )
    degree_factor = 1.0 / (
        1.0 + 0.12 * np.clip(degree_factor - 1.0, 0.0, n)
    )

    scores = (
        0.55 * direct_quality
        + 0.45 * shortcut_score
    )
    scores *= redundancy_penalty
    scores *= endpoint_balance
    scores *= degree_factor

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
