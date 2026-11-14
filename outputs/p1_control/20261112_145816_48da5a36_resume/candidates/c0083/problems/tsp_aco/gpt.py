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
        replacement = float(np.max(nonnegative)) if nonnegative.size else 1.0
        distances = np.where(finite, distances, replacement)

    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, np.inf)

    epsilon = np.finfo(float).eps
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, epsilon)

    # Quality-oriented signals.
    inverse_distance = 1.0 / np.maximum(distances, epsilon)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    inverse_max = float(np.max(inverse_distance))
    if inverse_max > 0.0:
        inverse_distance /= inverse_max

    row_sorted = np.sort(distances, axis=1)
    local_scale = row_sorted[:, 0]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        scale,
    )
    local_scale = np.maximum(local_scale, epsilon)

    relative_denominator = np.sqrt(
        np.outer(local_scale, local_scale)
    )
    relative_proximity = np.exp(-distances / relative_denominator)
    relative_proximity[~np.isfinite(relative_proximity)] = 0.0
    np.fill_diagonal(relative_proximity, 0.0)

    ranks = np.argsort(np.argsort(distances, axis=1), axis=1)
    rank_score = 1.0 / (1.0 + ranks.astype(float))
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        epsilon,
    )
    mutual_preference = np.sqrt(rank_score * rank_score.T)

    quality_signal = (
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
    quality_signal *= scale_consistency

    # Speed-oriented bounded signals.
    distance_signal = scale / (distances + scale)
    distance_signal[~np.isfinite(distance_signal)] = 0.0

    nearest = np.min(distances, axis=1)
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        scale,
    )
    nearest = np.maximum(nearest, epsilon)

    local_factor = 0.5 * (
        1.0 / nearest[:, None] + 1.0 / nearest[None, :]
    )
    local_signal = 1.0 / (1.0 + distances * local_factor)
    local_signal[~np.isfinite(local_signal)] = 0.0

    speed_signal = 0.75 * distance_signal + 0.25 * local_signal

    # Quality receives greater weight because its supplied objective is lower.
    score = 0.60 * quality_signal + 0.40 * speed_signal
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
