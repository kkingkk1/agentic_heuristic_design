import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    diagonal = np.eye(n, dtype=bool)
    forward_valid = np.isfinite(distances) & (distances >= 0.0) & ~diagonal
    reverse_distances = distances.T
    reverse_valid = (
        np.isfinite(reverse_distances)
        & (reverse_distances >= 0.0)
        & ~diagonal
    )

    distance = np.full((n, n), np.inf, dtype=float)

    both_valid = forward_valid & reverse_valid
    only_forward = forward_valid & ~reverse_valid
    only_reverse = reverse_valid & ~forward_valid

    distance[both_valid] = 0.5 * (
        distances[both_valid] + reverse_distances[both_valid]
    )
    distance[only_forward] = distances[only_forward]
    distance[only_reverse] = reverse_distances[only_reverse]

    np.fill_diagonal(distance, np.inf)
    valid = np.isfinite(distance)

    if not np.any(valid):
        return scores

    finite_values = distance[valid]
    positive_values = finite_values[finite_values > 0.0]
    eps = np.finfo(float).eps

    if positive_values.size:
        scale = float(np.median(positive_values))
    else:
        scale = max(float(np.median(finite_values)), 1.0)

    scale = max(scale, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        affinity = np.where(
            valid,
            1.0 / (1.0 + distance / scale),
            0.0,
        )

    degree = np.sum(affinity, axis=1)
    degree = np.maximum(degree, eps)

    degree_penalty = 1.0 / np.sqrt(
        degree[:, None] * degree[None, :]
    )
    degree_penalty /= max(float(np.max(degree_penalty)), eps)

    best_detour = np.full((n, n), np.inf, dtype=float)

    for intermediate in range(n):
        left = distance[:, intermediate][:, None]
        right = distance[intermediate, :][None, :]
        candidate = left + right
        best_detour = np.minimum(best_detour, candidate)

    np.fill_diagonal(best_detour, np.inf)

    detour_margin = np.zeros((n, n), dtype=float)
    usable_detours = valid & np.isfinite(best_detour)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        detour_margin[usable_detours] = (
            best_detour[usable_detours] - distance[usable_detours]
        ) / scale

    shortcut_bonus = 0.5 + 0.5 / (
        1.0 + np.exp(-np.clip(detour_margin, -20.0, 20.0))
    )
    shortcut_bonus[~usable_detours] = 0.5

    direct_affinity = np.zeros((n, n), dtype=float)
    direct_affinity[valid] = np.exp(
        -np.clip(distance[valid] / scale, 0.0, 50.0)
    )

    scores = (
        direct_affinity
        * shortcut_bonus
        * (0.55 + 0.45 * degree_penalty)
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
