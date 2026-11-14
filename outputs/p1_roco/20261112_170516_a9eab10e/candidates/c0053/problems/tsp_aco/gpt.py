import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return scale-invariant edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    with np.errstate(invalid="ignore", over="ignore"):
        d = 0.5 * (distances + distances.T)

    off_diagonal = ~np.eye(n, dtype=bool)
    valid = np.isfinite(d) & (d > 0) & off_diagonal
    positive = d[valid]

    if positive.size == 0:
        return np.zeros((n, n), dtype=float)

    fallback = float(np.max(positive))
    d = np.where(valid, d, fallback)
    np.fill_diagonal(d, np.inf)

    neighbor_count = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    nearest = np.partition(d, neighbor_count - 1, axis=1)[:, :neighbor_count]
    local_scale = np.mean(nearest, axis=1)
    local_scale = np.maximum(local_scale, np.finfo(float).eps)

    geometric_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :],
                   np.finfo(float).eps)
    )
    relative_length = d / geometric_scale

    visibility = 1.0 / (1.0 + relative_length)
    smooth_decay = np.exp(-0.9 * relative_length)
    long_edge_penalty = np.exp(
        -0.45 * np.maximum(relative_length - 1.0, 0.0)
    )

    ranks = np.argsort(np.argsort(d, axis=1), axis=1)
    rank_temperature = max(1.0, np.sqrt(float(n)))
    rank_strength = np.exp(-ranks.astype(float) / rank_temperature)

    mutual_bonus = np.sqrt(
        np.maximum(rank_strength * rank_strength.T, 0.0)
    )

    score = visibility * smooth_decay * long_edge_penalty
    score *= 1.0 + 0.9 * mutual_bonus
    score = 0.5 * (score + score.T)

    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score
