import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid = finite_nonnegative & finite_nonnegative.T
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric_distance = 0.5 * (distances + distances.T)
    symmetric_distance[~valid] = np.inf
    np.fill_diagonal(symmetric_distance, np.inf)

    positive = symmetric_distance[
        np.isfinite(symmetric_distance) & (symmetric_distance > 0.0)
    ]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    finite_distance = np.where(np.isfinite(symmetric_distance), symmetric_distance, 0.0)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (finite_distance[valid] + eps)

    row_max = np.max(inverse, axis=1, keepdims=True)
    relative_closeness = inverse / np.maximum(row_max, eps)

    temperature = max(0.90 * scale, eps)
    soft_closeness = np.zeros_like(inverse)
    soft_closeness[valid] = np.exp(
        -np.minimum(finite_distance[valid] / temperature, 700.0)
    )

    row_mass = soft_closeness.sum(axis=1)
    local_probability = soft_closeness / np.maximum(row_mass[:, None], eps)

    mutual_preference = np.sqrt(
        np.maximum(local_probability * local_probability.T, 0.0)
    )

    neighbor_count = valid.sum(axis=1).astype(float)
    weighted_degree = np.maximum(row_mass, eps)

    scarcity = 1.0 / np.sqrt(weighted_degree)
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.55, 1.90)

    endpoint_scarcity = np.sqrt(
        np.maximum(scarcity[:, None] * scarcity[None, :], 0.0)
    )

    degree_support = np.sqrt(
        np.maximum(
            neighbor_count[:, None] * neighbor_count[None, :],
            0.0,
        )
    )
    degree_support /= max(float(np.max(degree_support)), 1.0)

    score = inverse
    score *= 0.50 + 0.50 * relative_closeness
    score *= 0.72 + 0.48 * mutual_preference
    score *= 0.84 + 0.26 * endpoint_scarcity
    score *= 0.92 + 0.18 * degree_support
    score *= 0.88 + 0.12 * soft_closeness

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
