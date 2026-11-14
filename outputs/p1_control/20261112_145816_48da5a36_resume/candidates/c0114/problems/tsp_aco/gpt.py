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
    valid = distances[finite & (distances >= 0.0)]

    replacement = float(np.max(valid)) if valid.size else 1.0
    distances = np.where(
        finite & (distances >= 0.0),
        distances,
        replacement,
    )

    distances = 0.5 * (distances + distances.T)
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)

    eps = np.finfo(float).eps

    positive = distances[
        np.isfinite(distances) & (distances > 0.0)
    ]
    fallback = float(np.median(positive)) if positive.size else 1.0

    nearest = np.min(distances, axis=1)
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        fallback,
    )
    nearest = np.maximum(nearest, eps)

    local_scale = np.sqrt(nearest[:, None] * nearest[None, :])
    normalized_distance = distances / np.maximum(local_scale, eps)

    distance_score = 1.0 / np.maximum(distances, eps)
    distance_score[~np.isfinite(distance_score)] = 0.0

    adaptive_score = 1.0 / (1.0 + normalized_distance)
    adaptive_score[~np.isfinite(adaptive_score)] = 0.0

    ranks = np.argsort(
        np.argsort(distances, axis=1),
        axis=1,
    ).astype(float)

    rank_score = 1.0 / (1.0 + ranks)
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        eps,
    )

    mutual_rank = np.sqrt(rank_score * rank_score.T)

    row_max = np.max(distance_score, axis=1, keepdims=True)
    distance_score /= np.maximum(row_max, eps)

    score = (
        0.55 * distance_score
        + 0.30 * adaptive_score
        + 0.15 * mutual_rank
    )

    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
