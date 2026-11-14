import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros_like(distance_matrix, dtype=float)

    matrix = distance_matrix.copy()
    diagonal = np.eye(n, dtype=bool)

    finite_off_diagonal = np.isfinite(matrix) & ~diagonal
    if not np.any(finite_off_diagonal):
        return np.zeros_like(matrix, dtype=float)

    valid_values = matrix[finite_off_diagonal]
    fallback = float(np.max(valid_values))
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    matrix[~np.isfinite(matrix)] = fallback
    matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, np.inf)

    positive_values = matrix[np.isfinite(matrix) & (matrix > 0)]
    if positive_values.size == 0:
        scale = 1.0
    else:
        scale = float(np.median(positive_values))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.mean(positive_values))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

    distance_preference = np.exp(-matrix / scale)
    distance_preference[diagonal] = 0.0

    ranks = np.argsort(np.argsort(matrix, axis=1), axis=1).astype(float)
    ranks[diagonal] = np.inf

    local_competition = 1.0 / (1.0 + ranks)
    local_competition[diagonal] = 0.0

    candidate_count = max(2, int(np.ceil(np.sqrt(n))))
    candidate_mask = ranks < candidate_count
    candidate_mask[diagonal] = False

    candidate_density = np.sum(candidate_mask, axis=1).astype(float)
    scarcity = 1.0 / (1.0 + candidate_density)
    scarcity = scarcity / (np.mean(scarcity) + 1e-12)

    edge_scarcity = 0.5 * (scarcity[:, None] + scarcity[None, :])

    mutual_preference = np.sqrt(
        np.maximum(local_competition, 0.0)
        * np.maximum(local_competition.T, 0.0)
    )

    desirability = (
        0.50 * distance_preference
        + 0.30 * mutual_preference
        + 0.20 * edge_scarcity * candidate_mask.astype(float)
    )

    desirability = 0.5 * (desirability + desirability.T)
    desirability[diagonal] = 0.0
    desirability = np.maximum(desirability, 0.0)

    row_max = np.max(desirability, axis=1, keepdims=True)
    desirability = desirability / np.maximum(row_max, 1e-12)

    desirability[diagonal] = 0.0
    return desirability.astype(float)
