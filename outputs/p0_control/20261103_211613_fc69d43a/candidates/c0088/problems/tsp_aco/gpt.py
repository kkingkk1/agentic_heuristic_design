import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust local edge scores for a fixed ant-colony TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    positive_values = distances[valid]
    global_scale = float(np.median(positive_values))

    safe = np.where(valid, distances, global_scale)
    np.fill_diagonal(safe, np.nan)

    k = min(max(3, int(np.ceil(np.sqrt(n))),), n - 1)
    local_scales = np.empty(n, dtype=float)

    for i in range(n):
        row = safe[i]
        finite_row = row[np.isfinite(row) & (row > 0)]

        if finite_row.size == 0:
            local_scales[i] = global_scale
            continue

        nearest = np.partition(finite_row, min(k, finite_row.size) - 1)
        nearest = nearest[: min(k, finite_row.size)]
        local_scales[i] = float(np.median(nearest))

    local_scales = np.where(
        np.isfinite(local_scales) & (local_scales > 0),
        local_scales,
        global_scale,
    )

    log_distance = np.log(np.maximum(safe, np.finfo(float).tiny))
    log_scale = np.log(np.maximum(local_scales, np.finfo(float).tiny))

    endpoint_affinity = (
        0.5 * (log_scale[:, None] - log_distance)
        + 0.5 * (log_scale[None, :] - log_distance)
    )

    scale_compatibility = -0.5 * np.abs(
        log_scale[:, None] - log_scale[None, :]
    )

    log_scores = (
        -log_distance
        + 0.75 * endpoint_affinity
        + 0.25 * scale_compatibility
    )

    log_scores = np.where(valid, log_scores, -np.inf)

    finite_scores = log_scores[np.isfinite(log_scores)]
    if finite_scores.size == 0:
        return scores

    maximum = float(np.max(finite_scores))
    raw_scores = np.zeros_like(distances, dtype=float)
    raw_scores[valid] = np.exp(
        np.clip(log_scores[valid] - maximum, -745.0, 0.0)
    )

    valid_raw = raw_scores[valid]
    raw_maximum = float(np.max(valid_raw))

    if raw_maximum > 0:
        raw_scores /= raw_maximum

    np.fill_diagonal(raw_scores, 0.0)

    scores = 0.85 * raw_scores + 0.15 * raw_scores.T
    scores = np.where(valid | valid.T, scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
