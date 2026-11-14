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
    off_diagonal = ~np.eye(n, dtype=bool)
    valid = finite_nonnegative & off_diagonal
    symmetric_valid = valid & valid.T

    if not np.any(symmetric_valid):
        return np.zeros((n, n), dtype=float)

    forward = np.where(valid, distances, 0.0)
    backward = np.where(valid.T, distances.T, 0.0)
    symmetric_distances = 0.5 * (forward + backward)
    symmetric_distances[~symmetric_valid] = np.inf
    np.fill_diagonal(symmetric_distances, np.inf)

    finite_distances = symmetric_distances[np.isfinite(symmetric_distances)]
    positive_distances = finite_distances[finite_distances > 0.0]

    if positive_distances.size:
        scale = float(np.median(positive_distances))
    elif finite_distances.size:
        scale = float(np.mean(finite_distances))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps * max(1.0, scale)

    inverse_distance = 1.0 / (symmetric_distances + eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    np.fill_diagonal(inverse_distance, 0.0)

    max_inverse = np.max(inverse_distance, axis=1, keepdims=True)
    local_inverse = inverse_distance / np.maximum(max_inverse, eps)

    rank_order = np.argsort(
        np.where(symmetric_valid, symmetric_distances, np.inf),
        axis=1,
        kind="stable",
    )
    ranks = np.empty_like(rank_order, dtype=float)
    row_indices = np.arange(n)[:, None]
    ranks[row_indices, rank_order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~symmetric_valid] = 0.0

    temperature = max(0.75 * scale, eps)
    proximity = np.exp(
        -np.minimum(symmetric_distances, 60.0 * temperature) / temperature
    )
    proximity[~np.isfinite(proximity)] = 0.0
    np.fill_diagonal(proximity, 0.0)

    neighborhood_mass = proximity.sum(axis=1)
    normalized_proximity = proximity / np.maximum(
        neighborhood_mass[:, None], eps
    )

    mutual_preference = np.sqrt(
        np.maximum(normalized_proximity * normalized_proximity.T, 0.0)
    )

    top_k = max(1, min(n - 1, int(np.ceil(np.sqrt(n)))))
    top_mass = np.partition(
        proximity,
        kth=max(0, n - top_k),
        axis=1,
    )[:, -top_k:].sum(axis=1)

    scarcity = 1.0 / np.maximum(top_mass, eps)
    scarcity /= max(float(np.mean(scarcity)), eps)
    scarcity = np.clip(scarcity, 0.5, 2.5)

    endpoint_scarcity = np.sqrt(
        scarcity[:, None] * scarcity[None, :]
    )
    endpoint_scarcity = np.clip(endpoint_scarcity, 0.5, 2.5)

    reciprocal_rank = np.sqrt(
        np.maximum(rank_score * rank_score.T, 0.0)
    )

    distance_signal = np.sqrt(
        np.maximum(local_inverse * local_inverse.T, 0.0)
    )

    score = (
        0.34 * distance_signal
        + 0.24 * reciprocal_rank
        + 0.22 * mutual_preference
        + 0.12 * normalized_proximity
        + 0.08 * np.minimum(endpoint_scarcity / 2.5, 1.0)
    )

    score *= 0.75 + 0.25 * endpoint_scarcity
    score[~symmetric_valid] = 0.0

    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return score.astype(float, copy=False)
