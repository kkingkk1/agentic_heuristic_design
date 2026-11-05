import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2:
        raise ValueError("distance_matrix must be two-dimensional")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distance_matrix < 0):
        raise ValueError("distance_matrix must be non-negative")

    distances = 0.5 * (distance_matrix + distance_matrix.T)
    np.fill_diagonal(distances, np.inf)

    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    second_nearest = ordered[:, 1] if n > 2 else ordered[:, 0]

    positive = distances[np.isfinite(distances) & (distances > 0)]
    epsilon = np.finfo(float).eps
    global_scale = np.median(positive) if positive.size else 1.0
    global_scale = max(global_scale, epsilon)

    local_scale = 0.5 * (nearest + second_nearest)
    local_scale = np.maximum(local_scale, global_scale * 1e-12)

    relative_distance = distances / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    base_score = 1.0 / (relative_distance + epsilon)

    gap_i = (second_nearest[:, None] - distances) / (
        second_nearest[:, None] + epsilon
    )
    gap_j = (second_nearest[None, :] - distances) / (
        second_nearest[None, :] + epsilon
    )

    opportunity_bonus = 1.0 + 0.5 * np.clip(gap_i, -1.0, 1.0) * np.clip(
        gap_j, -1.0, 1.0
    )

    scores = base_score * opportunity_bonus
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores[~np.isfinite(scores)] = 0.0
    return scores
