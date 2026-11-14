import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = (
        np.isfinite(distances)
        & (distances >= 0.0)
        & ~diagonal
    )

    if not np.any(finite_nonnegative):
        return result

    forward = np.where(finite_nonnegative, distances, np.inf)
    backward = forward.T

    forward_ok = np.isfinite(forward)
    backward_ok = np.isfinite(backward)
    valid = forward_ok | backward_ok

    symmetric_distance = np.full((n, n), np.inf, dtype=float)
    both = forward_ok & backward_ok
    only_forward = forward_ok & ~backward_ok
    only_backward = backward_ok & ~forward_ok

    symmetric_distance[both] = (
        0.5 * forward[both] + 0.5 * backward[both]
    )
    symmetric_distance[only_forward] = forward[only_forward]
    symmetric_distance[only_backward] = backward[only_backward]
    np.fill_diagonal(symmetric_distance, np.inf)

    valid = np.isfinite(symmetric_distance) & ~diagonal
    if not np.any(valid):
        return result

    finite_values = symmetric_distance[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    else:
        fallback = float(np.median(finite_values))
        global_scale = max(fallback, 1.0)

    global_scale = max(global_scale, eps)

    # Estimate a robust local scale for each endpoint.
    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = symmetric_distance[i, valid[i]]
        positive_row = row[row > 0.0]

        if positive_row.size:
            local_scale[i] = float(
                np.percentile(positive_row, 60.0)
            )
        elif row.size:
            local_scale[i] = max(float(np.median(row)), eps)

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )

    safe_distance = np.where(
        valid,
        symmetric_distance,
        np.inf,
    )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        relative_distance = safe_distance / np.maximum(pair_scale, eps)

        affinity = np.exp(
            -np.clip(relative_distance, 0.0, 50.0)
        )

        # Soft endpoint-specific edge choice distributions.
        row_mass = np.sum(
            np.where(valid, affinity, 0.0),
            axis=1,
            keepdims=True,
        )
        row_mass = np.maximum(row_mass, eps)
        endpoint_choice = affinity / row_mass

        reverse_mass = np.sum(
            np.where(valid, affinity.T, 0.0),
            axis=1,
            keepdims=True,
        )
        reverse_mass = np.maximum(reverse_mass, eps)
        reverse_choice = affinity.T / reverse_mass

        mutual_support = np.sqrt(
            np.maximum(endpoint_choice, 0.0)
            * np.maximum(reverse_choice.T, 0.0)
        )

    # Compute the best two-hop replacement through an intermediate node.
    if n >= 3:
        first_leg = safe_distance[:, :, None]
        second_leg = safe_distance[None, :, :]

        two_hop_candidates = first_leg + second_leg

        endpoint_indices = np.arange(n)
        forbidden_intermediate = (
            endpoint_indices[None, :, None]
            == endpoint_indices[:, None, None]
        ) | (
            endpoint_indices[None, :, None]
            == endpoint_indices[None, None, :]
        )

        two_hop_candidates = np.where(
            forbidden_intermediate,
            np.inf,
            two_hop_candidates,
        )

        detour_distance = np.min(
            two_hop_candidates,
            axis=1,
        )
    else:
        detour_distance = np.full(
            (n, n),
            np.inf,
            dtype=float,
        )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        detour_gap = (
            detour_distance - safe_distance
        ) / np.maximum(pair_scale, global_scale)

    detour_gap = np.nan_to_num(
        detour_gap,
        nan=0.0,
        posinf=4.0,
        neginf=0.0,
    )

    irreplaceability = 1.0 + 0.28 * np.clip(
        detour_gap,
        0.0,
        4.0,
    )

    # Penalize edges that dominate both endpoint choice distributions.
    endpoint_load = np.sqrt(
        np.maximum(endpoint_choice, 0.0)
        * np.maximum(reverse_choice.T, 0.0)
    )
    load_penalty = 1.0 / (
        1.0 + 0.22 * np.clip(endpoint_load, 0.0, 4.0)
    )

    # Mildly prefer edges connecting endpoints with compatible local scales.
    scale_balance = np.minimum(
        local_scale[:, None],
        local_scale[None, :],
    ) / np.maximum(
        np.maximum(
            local_scale[:, None],
            local_scale[None, :],
        ),
        eps,
    )
    scale_balance = 0.90 + 0.10 * np.clip(
        scale_balance,
        0.0,
        1.0,
    )

    # Combine continuous signals without explicit rank or neighborhood masks.
    scores = (
        affinity
        * (0.62 + 0.38 * mutual_support)
        * irreplaceability
        * load_penalty
        * scale_balance
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
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
