import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, robust local-competition edge scores for TSP."""
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

    global_log_scale = np.log(global_scale)
    neighbor_limit = max(1, min(8, n - 1))
    log_scales = np.full(n, global_log_scale, dtype=float)
    valid_counts = valid.sum(axis=1)

    for i in range(n):
        row = distances[i, valid[i]]
        if row.size == 0:
            continue

        ordered = np.sort(row)
        k = min(neighbor_limit, ordered.size)

        log_nearest = np.log(max(float(ordered[0]), np.finfo(float).tiny))
        log_robust = np.log(
            max(float(np.quantile(ordered[:k], 0.60)), np.finfo(float).tiny)
        )
        log_local = 0.5 * (log_nearest + log_robust)

        reliability = min(1.0, max(0.0, (int(valid_counts[i]) - 1) / 7.0))
        anchor_weight = 0.05 + 0.15 * reliability

        log_scales[i] = (
            (1.0 - anchor_weight) * log_local
            + anchor_weight * global_log_scale
        )

    log_scores = np.full((n, n), -np.inf, dtype=float)
    maximum = -np.inf
    log_two = np.log(2.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n - 1):
            js = np.arange(i + 1, n)

            mutually_valid = valid[i, js] & valid[js, i]
            if not np.any(mutually_valid):
                continue

            js = js[mutually_valid]
            d_forward = distances[i, js]
            d_reverse = distances[js, i]

            log_scale_i = log_scales[i]
            log_scale_j = log_scales[js]

            log_ratio = np.minimum(log_scale_i, log_scale_j) - np.maximum(
                log_scale_i, log_scale_j
            )
            ratio = np.exp(log_ratio)

            log_compatibility = (
                log_two + log_ratio - np.log1p(ratio)
            )

            log_relative = 0.5 * (
                log_scale_i - np.log(d_forward)
                + log_scale_j - np.log(d_reverse)
            )

            log_penalty = -0.5 * (
                d_forward / np.exp(log_scale_i)
                + d_reverse / np.exp(log_scale_j)
            )

            raw = (
                0.80 * (log_relative + log_penalty)
                + 0.20 * log_compatibility
            )

            finite = np.isfinite(raw)
            if not np.any(finite):
                continue

            js = js[finite]
            raw = raw[finite]

            log_scores[i, js] = raw
            log_scores[js, i] = raw
            maximum = max(maximum, float(np.max(raw)))

    if not np.isfinite(maximum):
        return scores

    edge_mask = np.isfinite(log_scores)

    with np.errstate(over="ignore", invalid="ignore"):
        scores[edge_mask] = np.exp(
            np.clip(log_scores[edge_mask] - maximum, -745.0, 0.0)
        )

    np.fill_diagonal(scores, 0.0)
    return np.clip(scores, 0.0, 1.0).astype(float, copy=False)
