import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    if not np.all(finite):
        finite_values = distances[finite]
        nonnegative = finite_values[finite_values >= 0]
        replacement = (
            float(np.max(nonnegative)) if nonnegative.size else 1.0
        )
        distances = np.where(finite, distances, replacement)

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    positive = distances[
        np.isfinite(distances) & (distances > 0)
    ]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)
    epsilon = np.finfo(float).eps

    inverse_distance = 1.0 / np.maximum(distances, epsilon)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0

    maximum_inverse = float(np.max(inverse_distance))
    if maximum_inverse > 0:
        inverse_distance /= maximum_inverse

    row_sorted = np.sort(distances, axis=1)
    local_scale = row_sorted[:, 0]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        scale,
    )
    local_scale = np.maximum(local_scale, epsilon)

    relative_denominator = np.sqrt(
        np.outer(local_scale, local_scale)
    )
    relative_proximity = np.exp(
        -distances / relative_denominator
    )
    relative_proximity[~np.isfinite(relative_proximity)] = 0.0
    np.fill_diagonal(relative_proximity, 0.0)

    nearest_rank = np.argsort(
        np.argsort(distances, axis=1),
        axis=1,
    )
    rank_score = 1.0 / (1.0 + nearest_rank.astype(float))
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        epsilon,
    )
    mutual_preference = np.sqrt(rank_score * rank_score.T)

    base_score = (
        0.50 * inverse_distance
        + 0.30 * relative_proximity
        + 0.20 * mutual_preference
    )

    log_scales = np.log(local_scale)
    scale_difference = np.abs(
        log_scales[:, None] - log_scales[None, :]
    )
    scale_consistency = np.exp(-0.2 * scale_difference)
    scale_consistency = np.maximum(scale_consistency, 0.6)
    scale_consistency[~np.isfinite(scale_consistency)] = 0.6

    score = base_score * scale_consistency
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
