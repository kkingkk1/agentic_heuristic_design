import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge scores for a fixed ant-colony TSP solver.

    Higher scores favor edges that are short, locally competitive at both
    endpoints, and compatible with their shared neighborhoods.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    positive_values = distances[valid]
    global_scale = float(np.median(positive_values))
    if not np.isfinite(global_scale) or global_scale <= 0:
        return scores

    finite_counts = valid.sum(axis=1)
    neighbor_count = max(1, min(8, n - 1))

    local_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row_values = distances[i, valid[i]]
        if row_values.size == 0:
            continue

        ordered = np.sort(row_values)
        nearest = float(ordered[0])
        k = min(neighbor_count, ordered.size)
        robust_local = float(np.median(ordered[:k]))

        if np.isfinite(nearest) and nearest > 0 and np.isfinite(robust_local):
            local_scale[i] = np.sqrt(nearest * robust_local)

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    directed_affinity = np.zeros((n, n), dtype=float)
    eps = np.finfo(float).eps

    for i in range(n):
        valid_j = valid[i]
        if not np.any(valid_j):
            continue

        d = distances[i, valid_j]
        scale = local_scale[i]

        relative_shortness = scale / np.maximum(d, eps)
        locality = np.exp(-d / max(scale, eps))

        directed_affinity[i, valid_j] = (
            relative_shortness * locality
        )

    raw_scores = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if not (valid[i, j] or valid[j, i]):
                continue

            if valid[i, j] and valid[j, i]:
                d_ij = np.sqrt(
                    max(distances[i, j], eps) *
                    max(distances[j, i], eps)
                )

                bidirectional = np.sqrt(
                    max(directed_affinity[i, j], 0.0) *
                    max(directed_affinity[j, i], 0.0)
                )
            elif valid[i, j]:
                d_ij = distances[i, j]
                bidirectional = directed_affinity[i, j]
            else:
                d_ij = distances[j, i]
                bidirectional = directed_affinity[j, i]

            common = valid[i] & valid[j]
            common[i] = False
            common[j] = False

            if np.any(common):
                common_indices = np.flatnonzero(common)
                d_iq = distances[i, common_indices]
                d_jq = distances[j, common_indices]

                closeness_i = np.exp(
                    -d_iq / max(local_scale[i], eps)
                )
                closeness_j = np.exp(
                    -d_jq / max(local_scale[j], eps)
                )

                shared_weight = np.sqrt(closeness_i * closeness_j)
                compatibility = float(np.mean(shared_weight))

                neighborhood_support = min(
                    1.0,
                    common_indices.size /
                    max(1, min(finite_counts[i], finite_counts[j])),
                )
                compatibility = 0.5 * compatibility + 0.5 * neighborhood_support
            else:
                compatibility = 0.0

            absolute_shortness = global_scale / max(d_ij, eps)
            edge_value = (
                max(bidirectional, 0.0)
                * max(compatibility, 0.0)
                * absolute_shortness
            )

            if compatibility == 0.0:
                edge_value *= 0.5

            raw_scores[i, j] = edge_value
            raw_scores[j, i] = edge_value

    valid_undirected = np.triu(valid | valid.T, k=1)
    values = raw_scores[valid_undirected]

    if values.size == 0 or not np.any(np.isfinite(values)):
        return scores

    finite_values = values[np.isfinite(values) & (values > 0)]
    if finite_values.size == 0:
        return scores

    log_values = np.log(np.maximum(finite_values, eps))
    lower, upper = np.percentile(log_values, [5.0, 95.0])

    if not np.isfinite(lower) or not np.isfinite(upper):
        return scores

    if upper > lower:
        transformed = np.zeros_like(raw_scores)
        positive = valid & np.isfinite(raw_scores) & (raw_scores > 0)
        transformed[positive] = np.clip(
            (np.log(np.maximum(raw_scores[positive], eps)) - lower)
            / (upper - lower),
            0.0,
            1.0,
        )
        scores = transformed
    else:
        scores[valid] = 1.0

    scores = np.maximum(scores, 0.0)
    scores = np.minimum(scores, 1.0)
    np.fill_diagonal(scores, 0.0)

    return scores.astype(float, copy=False)
