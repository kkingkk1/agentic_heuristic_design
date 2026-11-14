import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(finite_nonnegative):
        return scores

    forward = np.where(finite_nonnegative, distances, np.inf)
    backward = forward.T

    forward_ok = np.isfinite(forward)
    backward_ok = np.isfinite(backward)

    both_ok = forward_ok & backward_ok
    only_forward = forward_ok & ~backward_ok
    only_backward = backward_ok & ~forward_ok

    edge_distance = np.full((n, n), np.inf, dtype=float)
    edge_distance[both_ok] = (
        forward[both_ok] + backward[both_ok]
    ) * 0.5
    edge_distance[only_forward] = forward[only_forward]
    edge_distance[only_backward] = backward[only_backward]

    np.fill_diagonal(edge_distance, np.inf)
    valid = np.isfinite(edge_distance)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    finite_values = edge_distance[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        global_scale = float(np.median(finite_values))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    safe_distance = np.where(valid, edge_distance, np.inf)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        conductance = np.where(
            valid,
            np.exp(-np.clip(safe_distance / global_scale, 0.0, 50.0)),
            0.0,
        )

    node_conductance = np.sum(conductance, axis=1)
    mean_conductance = float(np.mean(node_conductance))
    mean_conductance = max(mean_conductance, eps)

    endpoint_balance = np.sqrt(
        np.minimum(
            node_conductance[:, None] / mean_conductance,
            1.0,
        )
        * np.minimum(
            node_conductance[None, :] / mean_conductance,
            1.0,
        )
    )

    degree = np.sum(valid, axis=1).astype(float)
    degree_balance = np.sqrt(
        (1.0 + np.min(degree[:, None], axis=1)[:, None])
        / (1.0 + np.maximum(degree[:, None], 1.0))
    )
    degree_balance = np.sqrt(
        np.clip(
            (1.0 + np.minimum(degree[:, None], degree[None, :]))
            / (1.0 + np.maximum(degree[:, None], degree[None, :])),
            0.25,
            1.0,
        )
    )

    direct_kernel = np.exp(
        -np.clip(safe_distance / global_scale, 0.0, 50.0)
    )

    two_hop = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        through_k = (
            safe_distance[:, k, None]
            + safe_distance[None, k, :]
        )
        two_hop = np.minimum(two_hop, through_k)

    np.fill_diagonal(two_hop, np.inf)

    alternative_gap = np.zeros((n, n), dtype=float)
    usable_alternatives = valid & np.isfinite(two_hop)

    with np.errstate(divide="ignore", invalid="ignore"):
        alternative_gap[usable_alternatives] = np.clip(
            (
                two_hop[usable_alternatives]
                - safe_distance[usable_alternatives]
            )
            / global_scale,
            -4.0,
            8.0,
        )

    necessity = np.zeros((n, n), dtype=float)
    necessity[valid] = np.where(
        np.isfinite(two_hop[valid]),
        1.0 / (
            1.0
            + np.exp(-alternative_gap[valid])
        ),
        1.0,
    )

    local_best = np.min(
        np.where(valid, safe_distance, np.inf),
        axis=1,
    )
    local_best = np.where(
        np.isfinite(local_best),
        local_best,
        global_scale,
    )

    edge_relative_to_endpoints = safe_distance / np.maximum(
        np.sqrt(local_best[:, None] * local_best[None, :]),
        eps,
    )
    competitiveness = 1.0 / (
        1.0
        + 0.35 * np.clip(edge_relative_to_endpoints - 1.0, 0.0, 6.0)
    )

    scores = (
        direct_kernel
        * (0.72 + 0.28 * endpoint_balance)
        * (0.78 + 0.22 * degree_balance)
        * (0.70 + 0.30 * competitiveness)
        * (0.82 + 0.36 * necessity)
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
