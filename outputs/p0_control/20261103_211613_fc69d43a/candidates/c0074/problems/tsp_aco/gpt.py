import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric local-competition edge scores for a TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.full((n, n), np.nan, dtype=float)

    if n <= 1:
        return np.zeros((n, n), dtype=float)

    row_scales = np.full(n, np.nan, dtype=float)
    global_scale = np.inf

    for i in range(n):
        row = distances[i]
        valid = np.isfinite(row) & (row > 0)
        valid[i] = False

        if np.any(valid):
            local_scale = float(np.min(row[valid]))
            row_scales[i] = local_scale
            global_scale = min(global_scale, local_scale)

    if not np.isfinite(global_scale) or global_scale <= 0:
        return np.zeros((n, n), dtype=float)

    row_scales = np.where(
        np.isfinite(row_scales) & (row_scales > 0),
        row_scales,
        global_scale,
    )

    maximum_log_score = -np.inf
    found_valid_pair = False

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n - 1):
            indices = np.arange(i + 1, n)
            forward_distances = distances[i, indices]
            reverse_distances = distances[indices, i]

            valid = (
                np.isfinite(forward_distances)
                & (forward_distances > 0)
                & np.isfinite(reverse_distances)
                & (reverse_distances > 0)
            )

            if not np.any(valid):
                continue

            valid_indices = indices[valid]
            d_forward = forward_distances[valid]
            d_reverse = reverse_distances[valid]

            scale_i = row_scales[i]
            scale_j = row_scales[valid_indices]

            scale_min = np.minimum(scale_i, scale_j)
            scale_max = np.maximum(scale_i, scale_j)
            scale_ratio = np.divide(
                scale_min,
                scale_max,
                out=np.zeros_like(scale_min),
                where=scale_max > 0,
            )
            compatibility = 2.0 * scale_ratio / (1.0 + scale_ratio)

            directed_forward_log = np.log(scale_i) - np.log(d_forward)
            directed_reverse_log = np.log(scale_j) - np.log(d_reverse)

            raw_log_scores = (
                directed_forward_log
                + directed_reverse_log
                + np.log(compatibility)
            )

            finite_scores = np.isfinite(raw_log_scores)
            if not np.any(finite_scores):
                continue

            valid_indices = valid_indices[finite_scores]
            raw_log_scores = raw_log_scores[finite_scores]

            scores[i, valid_indices] = raw_log_scores
            scores[valid_indices, i] = raw_log_scores

            maximum_log_score = max(
                maximum_log_score,
                float(np.max(raw_log_scores)),
            )
            found_valid_pair = True

    if not found_valid_pair or not np.isfinite(maximum_log_score):
        return np.zeros((n, n), dtype=float)

    valid_scores = np.isfinite(scores)
    normalized = np.zeros_like(scores)
    with np.errstate(over="ignore", invalid="ignore"):
        normalized[valid_scores] = np.exp(
            scores[valid_scores] - maximum_log_score
        )

    np.fill_diagonal(normalized, 0.0)
    return np.clip(normalized, 0.0, 1.0).astype(float, copy=False)
