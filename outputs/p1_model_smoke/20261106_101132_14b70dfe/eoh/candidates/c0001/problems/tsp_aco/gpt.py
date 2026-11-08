import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite_distances = np.where(
        np.isfinite(distances) & (distances > 0),
        distances,
        np.nan,
    )

    positive_values = finite_distances[np.isfinite(finite_distances)]
    scale = float(np.median(positive_values)) if positive_values.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    proximity = np.zeros_like(distances, dtype=float)
    valid = np.isfinite(distances) & (distances > 0)
    proximity[valid] = 1.0 / (distances[valid] + scale * 1e-6)

    row_max = np.max(proximity, axis=1, keepdims=True)
    row_max[row_max <= 0] = 1.0
    local_competition = proximity / row_max

    row_sum = np.sum(proximity, axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    global_competition = proximity / row_sum

    score = proximity * (0.6 + 0.25 * local_competition + 0.15 * global_competition)
    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score
