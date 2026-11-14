import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric entropy-balanced edge indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0.0)
    np.fill_diagonal(positive, False)

    if not np.any(positive):
        result = np.ones((n, n), dtype=float)
        np.fill_diagonal(result, 0.0)
        return result

    positive_values = distances[positive]
    fallback = float(np.max(positive_values))
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    cleaned = np.where(finite, distances, fallback)
    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)
    np.fill_diagonal(cleaned, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    off_values = cleaned[off_diagonal]
    positive_off_values = off_values[off_values > 0.0]

    if positive_off_values.size:
        temperature = float(np.median(positive_off_values))
    else:
        temperature = 1.0

    eps = np.finfo(float).eps
    temperature = max(temperature, eps)

    base_log_weight = -np.clip(cleaned / temperature, 0.0, 60.0)
    base_log_weight[~off_diagonal] = -np.inf

    potentials = np.zeros(n, dtype=float)
    target_degree = 2.0

    for _ in range(250):
        log_weight = base_log_weight + potentials[:, None] + potentials[None, :]
        log_weight[~off_diagonal] = -np.inf

        weights = np.exp(np.clip(log_weight, -745.0, 0.0))
        row_sums = weights.sum(axis=1)

        valid_rows = row_sums > eps
        if not np.any(valid_rows):
            break

        correction = np.zeros(n, dtype=float)
        correction[valid_rows] = 0.5 * np.log(
            target_degree / row_sums[valid_rows]
        )
        potentials += correction

        if np.max(np.abs(correction[valid_rows])) < 1e-10:
            break

    log_weight = base_log_weight + potentials[:, None] + potentials[None, :]
    log_weight[~off_diagonal] = -np.inf

    score = np.exp(np.clip(log_weight, -745.0, 700.0))
    score = 0.5 * (score + score.T)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
