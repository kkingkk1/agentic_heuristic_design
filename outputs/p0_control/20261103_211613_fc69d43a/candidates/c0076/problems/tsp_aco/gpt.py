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

    if not np.any(valid):
        return scores

    positive = distances[valid]
    global_scale = float(np.median(positive))

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        return scores

    eps = np.finfo(float).tiny
    neighbor_limit = max(1, min(8, n - 1))
    local_scale = np.full(n, global_scale, dtype=float)
    valid_counts = valid.sum(axis=1)

    for i in range(n):
        row = distances[i, valid[i]]
        if row.size == 0:
            continue

        ordered = np.sort(row)
        k = min(neighbor_limit, ordered.size)

        nearest = max(float(ordered[0]), eps)
        robust_neighbor = max(
            float(np.quantile(ordered[:k], 0.60)),
            eps,
        )

        robust_scale = np.sqrt(nearest * robust_neighbor)

        if not np.isfinite(robust_scale) or robust_scale <= 0.0:
            continue

        # Weak log-domain shrinkage toward the global scale.
        # Sparse rows remain close to their nearest-edge estimate.
        count = int(valid_counts[i])
        reliability = min(1.0, max(0.0, (count - 1) / 7.0))
        anchor_weight = 0.05 + 0.15 * reliability

        local_scale[i] = np.exp(
            (1.0 - anchor_weight) * np.log(max(robust_scale, eps))
            + anchor_weight * np.log(max(global_scale, eps))
        )

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )

    log_directed = np.full((n, n), -np.inf, dtype=float)

    for i in range(n):
        js = np.flatnonzero(valid[i])
        if js.size == 0:
            continue

        scale = max(float(local_scale[i]), eps)
        d = np.maximum(distances[i, js], eps)

        # E-style endpoint competition in log space.
        log_directed[i, js] = (
            0.80 * np.log(scale / d)
            - d / scale
        )

    log_edges = np.full((n, n), -np.inf, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            # Preserve E's strict bidirectional validity rule.
            if not (valid[i, j] and valid[j, i]):
                continue

            value = 0.5 * (log_directed[i, j] + log_directed[j, i])

            if np.isfinite(value):
                log_edges[i, j] = value
                log_edges[j, i] = value

    upper_valid = np.triu(np.isfinite(log_edges), k=1)
    values = log_edges[upper_valid]

    if values.size == 0:
        return scores

    maximum = float(np.max(values))

    if not np.isfinite(maximum):
        return scores

    # Max normalization preserves the strongest edge as the reference.
    edge_mask = np.isfinite(log_edges)
    normalized = np.exp(np.clip(log_edges[edge_mask] - maximum, -745.0, 0.0))
    scores[edge_mask] = np.clip(normalized, 0.0, 1.0)

    scores = np.maximum(scores, scores.T)
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
