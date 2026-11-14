import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized symmetric edge priorities for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid_input = finite_nonnegative & ~diagonal

    if not np.any(valid_input):
        return np.zeros((n, n), dtype=float)

    # Build a symmetric distance view while retaining one-sided valid edges.
    forward = np.where(valid_input, distances, np.nan)
    backward = forward.T

    both = np.isfinite(forward) & np.isfinite(backward)
    only_forward = np.isfinite(forward) & ~np.isfinite(backward)
    only_backward = ~np.isfinite(forward) & np.isfinite(backward)

    safe = np.full((n, n), np.inf, dtype=float)
    safe[both] = 0.5 * (forward[both] + backward[both])
    safe[only_forward] = forward[only_forward]
    safe[only_backward] = backward[only_backward]

    valid = np.isfinite(safe) & ~diagonal
    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = safe[valid & (safe > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, np.finfo(float).eps)

    # Estimate a robust temperature for each endpoint from its own distances.
    row_temperatures = np.full(n, global_scale, dtype=float)
    for i in range(n):
        values = safe[i, valid[i]]
        if values.size:
            median = float(np.median(values))
            deviation = float(np.median(np.abs(values - median)))
            row_temperatures[i] = max(
                1.4826 * deviation + 0.25 * median,
                0.05 * global_scale,
                np.finfo(float).eps,
            )

    temperature = np.sqrt(
        row_temperatures[:, None] * row_temperatures[None, :]
    )

    with np.errstate(over="ignore", invalid="ignore"):
        affinity = np.exp(-safe / temperature)

    affinity[~valid] = 0.0

    # Row-normalized soft neighborhoods.
    row_mass = affinity.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        affinity,
        row_mass,
        out=np.zeros_like(affinity),
        where=row_mass > 0.0,
    )

    # Mutual soft compatibility of the two endpoint neighborhoods.
    mutual_affinity = np.sqrt(
        np.maximum(probabilities * probabilities.T, 0.0)
    )

    # Neighborhood-boundary contrast: favor edges joining different local regions.
    boundary = 0.5 * np.abs(probabilities - probabilities.T).sum(axis=1)
    boundary_pair = 0.5 * (boundary[:, None] + boundary[None, :])
    boundary_pair = np.clip(boundary_pair, 0.0, 1.0)

    # Measure whether an edge is a meaningful direct bridge compared with
    # the best two-hop route through any third node.
    detour = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if not valid[i, j]:
                continue

            candidates = valid[i] & valid[:, j]
            candidates[i] = False
            candidates[j] = False

            if np.any(candidates):
                best_two_hop = float(
                    np.min(safe[i, candidates] + safe[candidates, j])
                )
                excess = max(best_two_hop - safe[i, j], 0.0)
                detour_value = 1.0 - np.exp(-excess / global_scale)
            else:
                detour_value = 0.0

            detour[i, j] = detour_value
            detour[j, i] = detour_value

    # Mild absolute shortness prevents boundary and bridge signals from
    # selecting very long edges exclusively.
    shortness = 1.0 / (1.0 + (safe / global_scale) ** 2)
    shortness[~valid] = 0.0

    scores = (
        np.sqrt(np.maximum(mutual_affinity, 0.0))
        * (0.70 + 0.30 * shortness)
        * (0.75 + 0.25 * boundary_pair)
        * (0.85 + 0.35 * detour)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
