import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric local-competition edge scores for a TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    log_scales = np.full(n, np.nan, dtype=float)
    global_scale = np.inf

    for i in range(n):
        row = distances[i]
        valid = np.isfinite(row) & (row > 0)
        valid[i] = False

        if np.any(valid):
            local_scale = float(np.min(row[valid]))
            global_scale = min(global_scale, local_scale)
            log_scales[i] = np.log(local_scale)

    if not np.isfinite(global_scale) or global_scale <= 0:
        return np.zeros((n, n), dtype=float)

    global_log_scale = np.log(global_scale)
    log_scales = np.where(
        np.isfinite(log_scales),
        log_scales,
        global_log_scale,
    )

    log_scores = np.full((n, n), np.nan, dtype=float)
    maximum_log_score = -np.inf
    found_valid_pair = False
    log_two = np.log(2.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n - 1):
            start = i + 1
            forward_distances = distances[i, start:]
            reverse_distances = distances[start:, i]

            valid = (
                np.isfinite(forward_distances)
                & (forward_distances > 0)
                & np.isfinite(reverse_distances)
                & (reverse_distances > 0)
            )

            if not np.any(valid):
                continue

            positions = np.flatnonzero(valid)
            d_forward = forward_distances[positions]
            d_reverse = reverse_distances[positions]
            j = positions + start

            log_scale_i = log_scales[i]
            log_scale_j = log_scales[j]

            log_scale_min = np.minimum(log_scale_i, log_scale_j)
            log_scale_max = np.maximum(log_scale_i, log_scale_j)
            log_ratio = log_scale_min - log_scale_max
            ratio = np.exp(log_ratio)

            log_compatibility = (
                log_two
                + log_ratio
                - np.log1p(ratio)
            )

            raw_log_scores = (
                log_scale_i
                - np.log(d_forward)
                + log_scale_j
                - np.log(d_reverse)
                + log_compatibility
            )

            finite = np.isfinite(raw_log_scores)
            if not np.any(finite):
                continue

            j = j[finite]
            raw_log_scores = raw_log_scores[finite]

            log_scores[i, j] = raw_log_scores
            log_scores[j, i] = raw_log_scores

            maximum_log_score = max(
                maximum_log_score,
                float(np.max(raw_log_scores)),
            )
            found_valid_pair = True

    if not found_valid_pair or not np.isfinite(maximum_log_score):
        return np.zeros((n, n), dtype=float)

    valid_scores = np.isfinite(log_scores)
    scores = np.zeros_like(log_scores)

    with np.errstate(over="ignore", invalid="ignore"):
        scores[valid_scores] = np.exp(
            log_scores[valid_scores] - maximum_log_score
        )

    np.fill_diagonal(scores, 0.0)
    return np.clip(scores, 0.0, 1.0).astype(float, copy=False)
