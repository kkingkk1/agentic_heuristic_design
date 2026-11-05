import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric robust edge-score matrix for a fixed TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    positive = distances[valid]
    global_scale = float(np.median(positive))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        return scores

    eps = np.finfo(float).tiny
    log_scales = np.full(n, np.log(global_scale), dtype=float)

    for i in range(n):
        row = distances[i, valid[i]]
        if row.size == 0:
            continue

        ordered = np.sort(row)
        k = min(max(1, min(8, n - 1)), ordered.size)

        nearest = max(float(ordered[0]), eps)
        q60 = max(float(np.quantile(ordered[:k], 0.60)), eps)
        geometric_scale = np.sqrt(nearest * q60)
        median_scale = max(float(np.median(row)), eps)

        if not np.isfinite(geometric_scale) or geometric_scale <= 0.0:
            geometric_scale = median_scale

        count = int(row.size)
        density = count / max(1, n - 1)

        dispersion = abs(np.log(geometric_scale / median_scale))
        anomaly_weight = np.clip(dispersion / 2.0, 0.0, 1.0)

        local_weight = (
            0.72
            + 0.18 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))
            - 0.28 * anomaly_weight
        )
        local_weight = float(np.clip(local_weight, 0.35, 0.90))

        blended_scale = np.exp(
            local_weight * np.log(geometric_scale)
            + (1.0 - local_weight) * np.log(median_scale)
        )

        anchor_weight = 0.04 + 0.16 * np.sqrt(np.clip(density, 0.0, 1.0))
        log_scales[i] = (
            (1.0 - anchor_weight) * np.log(max(blended_scale, eps))
            + anchor_weight * np.log(max(global_scale, eps))
        )

    maximum_log_score = -np.inf

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n - 1):
            start = i + 1
            forward = distances[i, start:]
            reverse = distances[start:, i]

            edge_valid = (
                np.isfinite(forward)
                & (forward > 0.0)
                & np.isfinite(reverse)
                & (reverse > 0.0)
            )

            if not np.any(edge_valid):
                continue

            positions = np.flatnonzero(edge_valid)
            node_ids = positions + start
            forward = np.maximum(forward[positions], eps)
            reverse = np.maximum(reverse[positions], eps)

            relative_log = 0.5 * (
                np.log(forward) - log_scales[i]
                + np.log(reverse) - log_scales[node_ids]
            )
            normalized_distance = np.exp(
                np.clip(relative_log, -745.0, 709.0)
            )

            raw_log_scores = (
                -0.65 * relative_log
                -0.35 * normalized_distance
            )
            finite = np.isfinite(raw_log_scores)

            if np.any(finite):
                maximum_log_score = max(
                    maximum_log_score,
                    float(np.max(raw_log_scores[finite])),
                )

    if not np.isfinite(maximum_log_score):
        return scores

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n - 1):
            start = i + 1
            forward = distances[i, start:]
            reverse = distances[start:, i]

            edge_valid = (
                np.isfinite(forward)
                & (forward > 0.0)
                & np.isfinite(reverse)
                & (reverse > 0.0)
            )

            if not np.any(edge_valid):
                continue

            positions = np.flatnonzero(edge_valid)
            node_ids = positions + start
            forward = np.maximum(forward[positions], eps)
            reverse = np.maximum(reverse[positions], eps)

            relative_log = 0.5 * (
                np.log(forward) - log_scales[i]
                + np.log(reverse) - log_scales[node_ids]
            )
            normalized_distance = np.exp(
                np.clip(relative_log, -745.0, 709.0)
            )

            raw_log_scores = (
                -0.65 * relative_log
                -0.35 * normalized_distance
            )
            finite = np.isfinite(raw_log_scores)

            if not np.any(finite):
                continue

            node_ids = node_ids[finite]
            raw_log_scores = raw_log_scores[finite]
            edge_scores = np.exp(
                np.clip(raw_log_scores - maximum_log_score, -745.0, 0.0)
            )

            scores[i, node_ids] = edge_scores
            scores[node_ids, i] = edge_scores

    np.fill_diagonal(scores, 0.0)
    return np.clip(scores, 0.0, 1.0).astype(float, copy=False)
