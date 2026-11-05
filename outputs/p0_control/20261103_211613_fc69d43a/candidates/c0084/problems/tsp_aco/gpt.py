import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return scores

    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distance_matrix) & off_diagonal

    if not np.any(finite):
        return scores

    d = np.where(finite, np.maximum(distance_matrix, 0.0), np.nan)

    finite_values = d[finite]
    finite_max = np.max(finite_values)

    if not np.isfinite(finite_max):
        return scores

    positive_values = finite_values[finite_values > 0.0]
    global_positive_scale = (
        np.min(positive_values) if positive_values.size else 1.0
    )
    fallback_scale = max(global_positive_scale, np.finfo(float).eps)

    safe_distance = np.where(
        finite,
        np.maximum(d, 0.0),
        finite_max + fallback_scale,
    )

    k = min(3, n - 1)
    kth_index = k - 1

    row_scale = np.partition(
        np.where(finite, safe_distance, np.inf),
        kth_index,
        axis=1,
    )[:, kth_index]

    row_positive_min = np.min(
        np.where(finite & (safe_distance > 0.0), safe_distance, np.inf),
        axis=1,
    )

    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0.0),
        row_scale,
        row_positive_min,
    )
    row_scale = np.where(
        np.isfinite(row_scale) & (row_scale > 0.0),
        row_scale,
        fallback_scale,
    )
    row_scale = np.maximum(row_scale, np.finfo(float).eps)

    log_outgoing = -np.log1p(safe_distance / row_scale[:, None])
    log_incoming = -np.log1p(safe_distance / row_scale[None, :])

    log_bidirectional = 0.5 * (log_outgoing + log_incoming)
    bidirectional_affinity = np.exp(
        np.clip(log_bidirectional, -745.0, 0.0)
    )

    neighborhood = np.exp(
        np.clip(log_outgoing, -745.0, 0.0)
    )
    neighborhood[~finite] = 0.0
    np.fill_diagonal(neighborhood, 0.0)

    overlap = neighborhood @ neighborhood.T
    overlap_max = np.max(overlap, axis=1, keepdims=True)
    overlap_max = np.maximum(overlap_max, np.finfo(float).eps)
    overlap_compatibility = overlap / overlap_max

    log_scale = np.log(row_scale)
    scale_compatibility = np.exp(
        -np.abs(log_scale[:, None] - log_scale[None, :])
    )

    directed_scores = (
        bidirectional_affinity
        * (0.5 + 0.5 * overlap_compatibility)
        * (0.5 + 0.5 * scale_compatibility)
    )

    directed_scores[~finite] = 0.0
    np.fill_diagonal(directed_scores, 0.0)

    scores = 0.5 * (directed_scores + directed_scores.T)
    scores[~finite] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores
