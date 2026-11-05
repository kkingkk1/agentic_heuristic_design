import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric robust local-competition edge scores for a TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    log_scales = np.full(n, np.nan, dtype=float)

    for i in range(n):
        row = distances[i]
        valid = np.isfinite(row) & (row > 0)
        valid[i] = False

        if np.any(valid):
            local_scale = float(np.median(row[valid]))
            if np.isfinite(local_scale) and local_scale > 0:
                log_scales[i] = np.log(local_scale)

    finite_scales = np.isfinite(log_scales)
    if not np.any(finite_scales):
        return np.zeros((n, n), dtype=float)

    global_log_scale = float(np.median(log_scales[finite_scales]))
    log_scales = np.where(finite_scales, log_scales, global_log_scale)

    maximum_log_score = -np.inf

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n - 1):
            start = i + 1
            forward = distances[i, start:]
            reverse = distances[start:, i]

            valid = (
                np.isfinite(forward)
                & (forward > 0)
                & np.isfinite(reverse)
                & (reverse > 0)
            )

            if not np.any(valid):
                continue

            positions = np.flatnonzero(valid)
            forward = forward[positions]
            reverse = reverse[positions]
            node_ids = positions + start

            log_geometric_cost = 0.5 * (
                np.log(forward)
                - log_scales[i]
                + np.log(reverse)
                - log_scales[node_ids]
            )

            raw_log_scores = -np.exp(log_geometric_cost)
            finite = np.isfinite(raw_log_scores)

            if np.any(finite):
                maximum_log_score = max(
                    maximum_log_score,
                    float(np.max(raw_log_scores[finite])),
                )

    if not np.isfinite(maximum_log_score):
        return np.zeros((n, n), dtype=float)

    scores = np.zeros((n, n), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n - 1):
            start = i + 1
            forward = distances[i, start:]
            reverse = distances[start:, i]

            valid = (
                np.isfinite(forward)
                & (forward > 0)
                & np.isfinite(reverse)
                & (reverse > 0)
            )

            if not np.any(valid):
                continue

            positions = np.flatnonzero(valid)
            forward = forward[positions]
            reverse = reverse[positions]
            node_ids = positions + start

            log_geometric_cost = 0.5 * (
                np.log(forward)
                - log_scales[i]
                + np.log(reverse)
                - log_scales[node_ids]
            )

            raw_log_scores = -np.exp(log_geometric_cost)
            finite = np.isfinite(raw_log_scores)

            if not np.any(finite):
                continue

            node_ids = node_ids[finite]
            raw_log_scores = raw_log_scores[finite]
            edge_scores = np.exp(raw_log_scores - maximum_log_score)

            scores[i, node_ids] = edge_scores
            scores[node_ids, i] = edge_scores

    np.fill_diagonal(scores, 0.0)
    return np.clip(scores, 0.0, 1.0).astype(float, copy=False)
