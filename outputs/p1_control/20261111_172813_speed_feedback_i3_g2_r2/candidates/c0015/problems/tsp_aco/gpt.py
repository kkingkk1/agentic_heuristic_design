import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distance_matrix)
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite & off_diagonal

    positive = valid & (distance_matrix > 0)
    if np.any(positive):
        fallback = float(np.median(distance_matrix[positive]))
    else:
        fallback = 1.0

    distances = np.where(finite, distance_matrix, fallback)
    distances = np.maximum(distances, np.finfo(float).eps)
    np.fill_diagonal(distances, np.inf)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / distances[valid]

    row_max = np.max(inverse, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    preference = inverse / row_max
    preference[~valid] = 0.0

    k = min(n - 1, max(2, int(np.ceil(np.sqrt(n)))))

    masked_preference = np.where(valid, preference, -np.inf)
    selected = np.argpartition(
        -masked_preference,
        kth=k - 1,
        axis=1,
    )[:, :k]

    selected_values = np.take_along_axis(preference, selected, axis=1)
    selected_order = np.argsort(
        -selected_values,
        axis=1,
        kind="stable",
    )
    selected = np.take_along_axis(selected, selected_order, axis=1)

    selected_valid = np.take_along_axis(valid, selected, axis=1)
    rank_weights = 1.0 / np.arange(1, k + 1, dtype=float)
    selected_scores = selected_valid * rank_weights[None, :]

    scarcity = np.zeros((n, n), dtype=float)
    rows = np.broadcast_to(np.arange(n)[:, None], selected.shape)
    scarcity[rows, selected] = selected_scores

    scarcity = scarcity + scarcity.T
    scarcity_max = np.max(scarcity)
    if scarcity_max > 0:
        scarcity /= scarcity_max

    score = 0.70 * preference + 0.30 * scarcity
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
