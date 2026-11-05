import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square two-dimensional array")

    n = distance_matrix.shape[0]
    scores = np.zeros_like(distance_matrix, dtype=float)

    if n <= 1:
        return scores

    if np.any(np.isfinite(distance_matrix) & (distance_matrix < 0)):
        raise ValueError("distance_matrix cannot contain negative distances")

    valid = np.isfinite(distance_matrix)
    np.fill_diagonal(valid, False)

    positive_edges = valid & (distance_matrix > 0)
    positive_distances = distance_matrix[positive_edges]

    if positive_distances.size == 0:
        return scores

    global_scale = float(np.median(positive_distances))
    epsilon = max(global_scale * 1e-12, np.finfo(float).eps)

    local_scales = np.empty(n, dtype=float)
    neighbor_count = max(1, int(np.ceil(np.sqrt(n - 1))))

    for i in range(n):
        candidates = distance_matrix[i][positive_edges[i]]

        if candidates.size == 0:
            local_scales[i] = global_scale
            continue

        nearest = np.partition(
            candidates,
            min(neighbor_count, candidates.size) - 1,
        )[: min(neighbor_count, candidates.size)]

        local_scales[i] = float(np.mean(nearest))

    local_scales = np.maximum(local_scales, epsilon)

    endpoint_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )
    normalized_distance = distance_matrix / np.maximum(endpoint_scale, epsilon)

    finite_edges = positive_edges & np.isfinite(normalized_distance)

    edge_scores = np.zeros_like(distance_matrix, dtype=float)
    normalized_edges = normalized_distance[finite_edges]

    decay_rate = 0.9
    reciprocal_weight = 1.0

    edge_scores[finite_edges] = (
        np.exp(-decay_rate * normalized_edges)
        / (1.0 + reciprocal_weight * normalized_edges)
    )

    edge_scores = 0.5 * (edge_scores + edge_scores.T)
    np.fill_diagonal(edge_scores, 0.0)

    return edge_scores
