import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-robust heuristic scores for a TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    bidirectional = valid & valid.T
    if not np.any(bidirectional):
        return scores

    positive = distances[valid]
    global_scale = float(np.median(positive))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        return scores

    eps = np.finfo(float).tiny
    neighbor_limit = max(1, min(8, n - 1))
    local_scales = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row = distances[i, valid[i]]
        if row.size == 0:
            continue

        ordered = np.sort(row)
        k = min(neighbor_limit, ordered.size)

        nearest = max(float(ordered[0]), eps)
        neighborhood = max(
            float(np.quantile(ordered[:k], 0.60)),
            eps,
        )

        robust_scale = np.sqrt(nearest * neighborhood)
        if not np.isfinite(robust_scale) or robust_scale <= 0.0:
            continue

        reliability = min(1.0, max(0.0, (row.size - 1) / 7.0))
        anchor_weight = 0.05 + 0.15 * reliability

        local_scales[i] = np.exp(
            (1.0 - anchor_weight) * np.log(robust_scale)
            + anchor_weight * np.log(max(global_scale, eps))
        )

    local_scales = np.where(
        np.isfinite(local_scales) & (local_scales > 0.0),
        local_scales,
        global_scale,
    )

    log_scores = np.full((n, n), -np.inf, dtype=float)
    maximum = -np.inf

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for i in range(n - 1):
            js = np.arange(i + 1, n)
            pair_valid = bidirectional[i, js]

            if not np.any(pair_valid):
                continue

            js = js[pair_valid]
            d_forward = np.maximum(distances[i, js], eps)
            d_reverse = np.maximum(distances[js, i], eps)

            scale_i = max(float(local_scales[i]), eps)
            scale_j = np.maximum(local_scales[js], eps)

            ratio = np.minimum(scale_i, scale_j) / np.maximum(
                scale_i,
                scale_j,
            )
            compatibility = 2.0 * ratio / (1.0 + ratio)

            directed_forward = (
                0.80 * (np.log(scale_i) - np.log(d_forward))
                - d_forward / scale_i
            )
            directed_reverse = (
                0.80 * (np.log(scale_j) - np.log(d_reverse))
                - d_reverse / scale_j
            )

            raw = (
                0.5 * (directed_forward + directed_reverse)
                + np.log(np.maximum(compatibility, eps))
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
        normalized = np.exp(
            np.clip(log_scores[edge_mask] - maximum, -745.0, 0.0)
        )

    scores[edge_mask] = np.clip(normalized, 0.0, 1.0)
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
