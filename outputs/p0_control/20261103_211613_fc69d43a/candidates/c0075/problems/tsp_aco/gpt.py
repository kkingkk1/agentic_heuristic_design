import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric, scale-robust heuristic scores for a TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.full((n, n), -np.inf, dtype=float)

    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = distances[valid]
    global_scale = float(np.median(positive))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).tiny
    neighbor_count = min(8, n - 1)
    local_scales = np.full(n, global_scale, dtype=float)
    valid_counts = valid.sum(axis=1)

    for i in range(n):
        row = distances[i, valid[i]]
        if row.size == 0:
            continue

        ordered = np.sort(row)
        closest = ordered[: min(neighbor_count, ordered.size)]
        nearest = max(float(closest[0]), eps)
        quantile = max(float(np.quantile(closest, 0.60)), eps)
        estimate = np.sqrt(nearest * quantile)

        if np.isfinite(estimate) and estimate > 0.0:
            local_scales[i] = estimate

    local_scales = np.sqrt(
        np.maximum(local_scales, eps) * max(global_scale, eps)
    )
    local_scales = np.where(
        np.isfinite(local_scales) & (local_scales > 0.0),
        local_scales,
        global_scale,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_directed = np.full((n, n), -np.inf, dtype=float)

        for i in range(n):
            indices = np.flatnonzero(valid[i])
            if indices.size == 0:
                continue

            edge_distances = np.maximum(distances[i, indices], eps)
            scale = max(float(local_scales[i]), eps)

            log_directed[i, indices] = (
                0.80 * (np.log(scale) - np.log(edge_distances))
                - edge_distances / scale
            )

        for i in range(n - 1):
            for j in range(i + 1, n):
                ij = bool(valid[i, j])
                ji = bool(valid[j, i])

                if not (ij or ji):
                    continue

                if ij and ji:
                    endpoint_log = 0.5 * (
                        log_directed[i, j] + log_directed[j, i]
                    )
                    edge_distance = np.sqrt(
                        max(distances[i, j], eps)
                        * max(distances[j, i], eps)
                    )
                elif ij:
                    endpoint_log = log_directed[i, j]
                    edge_distance = max(distances[i, j], eps)
                else:
                    endpoint_log = log_directed[j, i]
                    edge_distance = max(distances[j, i], eps)

                if not np.isfinite(endpoint_log):
                    continue

                common = valid[i] & valid[j]
                common[i] = False
                common[j] = False
                common_indices = np.flatnonzero(common)

                if common_indices.size:
                    di = np.maximum(distances[i, common_indices], eps)
                    dj = np.maximum(distances[j, common_indices], eps)

                    shared_log_closeness = -0.5 * (
                        di / max(local_scales[i], eps)
                        + dj / max(local_scales[j], eps)
                    )

                    keep = min(4, shared_log_closeness.size)
                    strongest = np.partition(
                        shared_log_closeness, -keep
                    )[-keep:]
                    shared_closeness = float(np.exp(np.mean(strongest)))

                    denominator = max(
                        1,
                        min(
                            int(valid_counts[i]),
                            int(valid_counts[j]),
                        ),
                    )
                    support = common_indices.size / denominator
                    compatibility = 0.75 + 0.25 * (
                        0.65 * np.clip(shared_closeness, 0.0, 1.0)
                        + 0.35 * np.clip(support, 0.0, 1.0)
                    )
                else:
                    compatibility = 0.75

                global_term = (
                    max(global_scale, eps)
                    / max(edge_distance, eps)
                ) ** 0.30

                log_score = (
                    endpoint_log
                    + np.log(max(compatibility, eps))
                    + np.log(max(global_term, eps))
                )

                if np.isfinite(log_score):
                    scores[i, j] = log_score
                    scores[j, i] = log_score

    valid_scores = np.isfinite(scores)
    if not np.any(valid_scores):
        return np.zeros((n, n), dtype=float)

    maximum = float(np.max(scores[valid_scores]))
    normalized = np.zeros((n, n), dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        normalized[valid_scores] = np.exp(
            scores[valid_scores] - maximum
        )

    np.fill_diagonal(normalized, 0.0)
    return np.clip(normalized, 0.0, 1.0).astype(float, copy=False)
