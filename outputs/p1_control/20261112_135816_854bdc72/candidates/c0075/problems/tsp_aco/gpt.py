import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        finite_values = distances[finite]
        nonnegative = finite_values[finite_values >= 0.0]
        replacement = (
            float(np.max(nonnegative))
            if nonnegative.size
            else 1.0
        )
        distances = np.where(finite, distances, replacement)

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    eps = np.finfo(float).eps
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    fallback = max(fallback, eps)

    nearest = np.min(distances, axis=1)
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        fallback,
    )
    nearest = np.maximum(nearest, eps)

    inverse_distance = 1.0 / np.maximum(distances, eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    inverse_max = float(np.max(inverse_distance))
    if inverse_max > 0.0:
        inverse_distance /= inverse_max

    geometric_scale = np.sqrt(
        np.outer(nearest, nearest)
    )

    relative_proximity = np.exp(
        -distances / geometric_scale
    )
    relative_proximity[~np.isfinite(relative_proximity)] = 0.0

    local_closeness = np.exp(
        -0.5
        * distances
        * (
            1.0 / nearest[:, None]
            + 1.0 / nearest[None, :]
        )
    )
    local_closeness[~np.isfinite(local_closeness)] = 0.0

    base_score = (
        0.55 * inverse_distance
        + 0.25 * relative_proximity
        + 0.20 * local_closeness
    )

    log_scales = np.log(nearest)
    scale_difference = np.abs(
        log_scales[:, None] - log_scales[None, :]
    )
    scale_consistency = np.exp(-0.2 * scale_difference)
    scale_consistency[~np.isfinite(scale_consistency)] = 0.0

    score = base_score * scale_consistency
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
