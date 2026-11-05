import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric adaptive edge-score matrix for a minimization TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0.0):
        raise ValueError("distance_matrix must contain non-negative values")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    d = 0.5 * (distances + distances.T)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, np.inf)

    positive = d[np.isfinite(d) & (d > 0.0)]
    if positive.size == 0:
        return scores

    global_scale = max(
        float(np.median(positive)),
        np.finfo(float).eps,
    )

    local_scales = np.empty(n, dtype=float)
    for i in range(n):
        row = d[i]
        candidates = row[np.isfinite(row) & (row > 0.0)]
        local_scales[i] = (
            float(np.median(candidates))
            if candidates.size
            else global_scale
        )

    local_scales = np.maximum(local_scales, np.finfo(float).eps)
    endpoint_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )

    relative_length = d / endpoint_scale
    finite_edges = np.isfinite(d)

    direct = np.zeros((n, n), dtype=float)
    direct[finite_edges] = np.exp(
        -np.minimum(relative_length[finite_edges], 700.0)
    )

    source_preference = np.zeros((n, n), dtype=float)
    source_preference[finite_edges] = np.exp(
        -np.minimum(d[finite_edges] / local_scales[:, None][
            np.broadcast_to(np.arange(n)[:, None], (n, n))
        ], 700.0)
    )

    target_preference = np.zeros((n, n), dtype=float)
    target_preference[finite_edges] = np.exp(
        -np.minimum(d[finite_edges] / local_scales[None, :], 700.0)
    )

    competition = np.sqrt(source_preference * target_preference)

    neighborhood = source_preference.copy()
    np.fill_diagonal(neighborhood, 0.0)
    overlap = neighborhood @ neighborhood.T
    overlap_max = np.max(overlap, axis=1, keepdims=True)
    overlap_max = np.maximum(overlap_max, np.finfo(float).eps)
    compatibility = overlap / overlap_max

    through = d[:, :, None] + d[None, :, :]
    best_two_hop = np.min(through, axis=1)
    bridge_ratio = np.zeros((n, n), dtype=float)
    valid_ratio = finite_edges & np.isfinite(best_two_hop)
    bridge_ratio[valid_ratio] = (
        best_two_hop[valid_ratio]
        / np.maximum(d[valid_ratio], np.finfo(float).eps)
    )
    bridge_bonus = np.zeros((n, n), dtype=float)
    bridge_bonus[valid_ratio] = 1.0 / (
        1.0 + np.maximum(bridge_ratio[valid_ratio], 0.0)
    )

    scores = direct * (0.5 + 0.5 * competition)
    scores *= 0.75 + 0.25 * compatibility
    scores *= 1.0 + 0.35 * bridge_bonus

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)
    scores[~np.isfinite(scores)] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
