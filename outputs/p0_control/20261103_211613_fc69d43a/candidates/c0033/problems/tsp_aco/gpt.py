import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a directed regret-and-shortcut heuristic matrix for TSP."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite_negative = np.isfinite(distances) & (distances < 0.0)
    if np.any(finite_negative):
        raise ValueError("distance_matrix must contain non-negative distances")

    valid = np.isfinite(distances)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    safe = np.where(valid, distances, np.inf)

    finite_values = safe[valid]
    positive_values = finite_values[finite_values > 0.0]
    distance_scale = (
        float(np.median(positive_values))
        if positive_values.size
        else 1.0
    )
    eps = np.finfo(float).eps * max(distance_scale, 1.0)

    row_counts = np.sum(valid, axis=1)
    col_counts = np.sum(valid, axis=0)

    row_order = np.argsort(np.where(valid, safe, np.inf), axis=1)
    col_order = np.argsort(np.where(valid, safe, np.inf), axis=0)

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)
    row_rank.fill(1.0)
    col_rank.fill(1.0)

    for i in range(n):
        ordered = row_order[i]
        valid_positions = np.flatnonzero(valid[i, ordered])
        if valid_positions.size:
            row_rank[i, ordered[valid_positions]] = (
                valid_positions + 1.0
            ) / valid_positions.size

    for j in range(n):
        ordered = col_order[:, j]
        valid_positions = np.flatnonzero(valid[ordered, j])
        if valid_positions.size:
            col_rank[ordered[valid_positions], j] = (
                valid_positions + 1.0
            ) / valid_positions.size

    row_preference = np.maximum(1.0 - row_rank, 0.0)
    col_preference = np.maximum(1.0 - col_rank, 0.0)
    rank_signal = np.sqrt(row_preference * col_preference)

    if n >= 3:
        two_hop = np.min(
            safe[:, :, None] + safe[None, :, :],
            axis=1,
        )

        direct = np.where(valid, safe, np.inf)
        detour_gap = two_hop - direct

        finite_gaps = detour_gap[valid & np.isfinite(detour_gap)]
        gap_scale = (
            float(np.median(np.abs(finite_gaps)))
            if finite_gaps.size
            else distance_scale
        )
        gap_scale = max(gap_scale, eps)

        shortcut_signal = 1.0 / (
            1.0 + np.exp(-np.clip(detour_gap / gap_scale, -60.0, 60.0))
        )
    else:
        shortcut_signal = np.full((n, n), 0.5, dtype=float)

    magnitude_signal = 1.0 / np.maximum(safe / distance_scale, eps)
    magnitude_signal = magnitude_signal / (
        1.0 + magnitude_signal
    )

    combined = (
        np.sqrt(np.maximum(rank_signal, 0.0))
        * (0.5 + 0.5 * shortcut_signal)
        * (0.5 + 0.5 * magnitude_signal)
    )

    scores[valid] = combined[valid]
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
