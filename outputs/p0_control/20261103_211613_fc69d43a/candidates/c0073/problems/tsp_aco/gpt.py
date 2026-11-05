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
    neighbor_count = max(1, min(8, n - 1))
    local_scale = np.full(n, global_scale, dtype=float)
    finite_counts = valid.sum(axis=1)

    for i in range(n):
        row = distances[i, valid[i]]
        if row.size == 0:
            continue

        ordered = np.sort(row)
        k = min(neighbor_count, ordered.size)
        nearest = max(float(ordered[0]), eps)
        local_quantile = max(float(np.quantile(ordered[:k], 0.60)), eps)

        adaptive = np.sqrt(nearest * local_quantile)
        if np.isfinite(adaptive) and adaptive > 0.0:
            local_scale[i] = adaptive

    local_scale = np.sqrt(
        np.maximum(local_scale, eps) * max(global_scale, eps)
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

        d = np.maximum(distances[i, js], eps)
        scale = max(float(local_scale[i]), eps)
        relative = scale / d

        # Monotone short-edge preference with bounded numerical behavior.
        log_directed[i, js] = (
            0.80 * np.log(np.maximum(relative, eps))
            - d / scale
        )

    log_raw = np.full((n, n), -np.inf, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            ij = bool(valid[i, j])
            ji = bool(valid[j, i])

            if not (ij or ji):
                continue

            if ij and ji:
                endpoint_log_affinity = 0.5 * (
                    log_directed[i, j] + log_directed[j, i]
                )
                edge_distance = np.sqrt(
                    max(distances[i, j], eps)
                    * max(distances[j, i], eps)
                )
            elif ij:
                endpoint_log_affinity = log_directed[i, j]
                edge_distance = max(distances[i, j], eps)
            else:
                endpoint_log_affinity = log_directed[j, i]
                edge_distance = max(distances[j, i], eps)

            common = valid[i] & valid[j]
            common[i] = False
            common[j] = False
            common_indices = np.flatnonzero(common)

            if common_indices.size:
                di = np.maximum(distances[i, common_indices], eps)
                dj = np.maximum(distances[j, common_indices], eps)

                shared_log_closeness = -0.5 * (
                    di / max(local_scale[i], eps)
                    + dj / max(local_scale[j], eps)
                )

                # Strong shared neighbors carry more information than distant ones.
                keep = min(4, shared_log_closeness.size)
                strongest = np.partition(
                    shared_log_closeness, -keep
                )[-keep:]
                shared_closeness = float(np.exp(np.mean(strongest)))

                support = common_indices.size / max(
                    1, min(int(finite_counts[i]), int(finite_counts[j]))
                )
                compatibility = 0.75 + 0.25 * (
                    0.65 * np.clip(shared_closeness, 0.0, 1.0)
                    + 0.35 * np.clip(support, 0.0, 1.0)
                )
            else:
                # Preserve strong isolated edges while giving connected
                # neighborhoods a modest advantage.
                compatibility = 0.75

            global_term = (
                max(global_scale, eps) / max(edge_distance, eps)
            ) ** 0.30

            log_value = (
                endpoint_log_affinity
                + np.log(max(compatibility, eps))
                + np.log(max(global_term, eps))
            )

            if np.isfinite(log_value):
                log_raw[i, j] = log_value
                log_raw[j, i] = log_value

    undirected_valid = np.triu(valid | valid.T, k=1)
    values = log_raw[undirected_valid]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return scores

    lower, upper = np.percentile(values, [5.0, 95.0])
    if not np.isfinite(lower) or not np.isfinite(upper):
        return scores

    output_valid = (valid | valid.T) & np.isfinite(log_raw)

    if upper > lower:
        normalized = (log_raw[output_valid] - lower) / (upper - lower)
        scores[output_valid] = np.clip(normalized, 0.0, 1.0)
    else:
        scores[output_valid] = 1.0

    scores = np.clip(scores, 0.0, 1.0)
    scores = np.maximum(scores, scores.T)
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
