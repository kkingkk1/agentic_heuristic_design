import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    d = 0.5 * (distances + distances.T)
    valid = np.isfinite(d) & (d > 0)
    positive_values = d[valid]

    if positive_values.size == 0:
        return np.zeros_like(d, dtype=float)

    fallback = float(np.max(positive_values))
    d = np.where(valid, d, fallback)
    np.fill_diagonal(d, np.inf)

    local_scale = np.min(d, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    geometric_scale = np.sqrt(
        np.maximum(
            local_scale[:, None] * local_scale[None, :],
            np.finfo(float).eps,
        )
    )

    relative_length = d / geometric_scale
    visibility = geometric_scale / np.maximum(d, np.finfo(float).eps)

    proximity = 1.0 / (1.0 + 0.85 * relative_length)
    long_edge_penalty = np.exp(
        -0.9 * np.maximum(relative_length - 1.0, 0.0)
    )

    ranks = np.argsort(np.argsort(d, axis=1), axis=1)
    rank_temperature = max(1.0, np.sqrt(float(n)) * 0.9)
    neighbor_strength = np.exp(
        -ranks.astype(float) / rank_temperature
    )

    mutual_neighbor_bonus = np.sqrt(
        neighbor_strength * neighbor_strength.T
    )

    score = visibility * proximity * long_edge_penalty
    score *= 0.65 + 0.95 * mutual_neighbor_bonus

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0:
        score /= maximum

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
