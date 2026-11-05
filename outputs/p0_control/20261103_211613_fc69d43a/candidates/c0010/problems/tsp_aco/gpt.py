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

    positive_distances = distance_matrix[valid & (distance_matrix > 0)]
    if positive_distances.size == 0:
        return scores

    global_scale = float(np.median(positive_distances))
    epsilon = max(global_scale * 1e-12, np.finfo(float).eps)

    local_scales = np.empty(n, dtype=float)
    neighbor_count = max(1, int(np.ceil(np.sqrt(n - 1))))

    for i in range(n):
        row = distance_matrix[i]
        candidates = row[valid[i] & (row > 0)]

        if candidates.size == 0:
            local_scales[i] = global_scale
            continue

        candidates = np.sort(candidates)
        local_scales[i] = float(
            np.mean(candidates[: min(neighbor_count, candidates.size)])
        )

    local_scales = np.maximum(local_scales, epsilon)

    endpoint_scale = np.sqrt(
        local_scales[:, None] * local_scales[None, :]
    )
    normalized_distance = distance_matrix / np.maximum(endpoint_scale, epsilon)

    finite_edges = valid & np.isfinite(normalized_distance)
    edge_score = np.zeros_like(distance_matrix, dtype=float)

    edge_score[finite_edges] = (
        np.exp(-normalized_distance[finite_edges])
        / (1.0 + normalized_distance[finite_edges])
    )

    edge_score = 0.5 * (edge_score + edge_score.T)
    np.fill_diagonal(edge_score, 0.0)

    return edge_score
