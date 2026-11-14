import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    finite = np.isfinite(distances)
    if not np.any(finite):
        return np.zeros_like(distances, dtype=float)

    valid_positive = distances[finite & (distances > 0.0)]
    scale = float(np.median(valid_positive)) if valid_positive.size else 1.0
    scale = max(scale, np.finfo(float).tiny)

    safe_distances = np.where(
        finite,
        np.maximum(distances, 0.0),
        np.inf,
    )
    np.fill_diagonal(safe_distances, np.inf)

    positive_edges = np.isfinite(safe_distances) & (safe_distances > 0.0)
    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)

    normalized_distance = safe_distances / scale
    distance_score = np.exp(-np.minimum(normalized_distance, 60.0))
    distance_score[~positive_edges] = 0.0

    inverse_distance = 1.0 / (safe_distances + eps)
    inverse_distance[~positive_edges] = 0.0

    row_max = np.max(inverse_distance, axis=1, keepdims=True)
    col_max = np.max(inverse_distance, axis=0, keepdims=True)

    row_relative = inverse_distance / (row_max + eps)
    col_relative = inverse_distance / (col_max + eps)

    row_rank = np.argsort(np.argsort(-inverse_distance, axis=1), axis=1)
    col_rank = np.argsort(np.argsort(-inverse_distance, axis=0), axis=0)

    candidate_width = max(1, min(n - 1, int(np.ceil(np.sqrt(max(n, 1))))))
    row_rank_score = np.exp(-row_rank / candidate_width)
    col_rank_score = np.exp(-col_rank / candidate_width)
    row_rank_score[~positive_edges] = 0.0
    col_rank_score[~positive_edges] = 0.0

    mutual_support = np.sqrt(
        np.maximum(row_relative, 0.0) * np.maximum(col_relative, 0.0)
    )
    rank_support = np.sqrt(row_rank_score * col_rank_score)

    priors = (
        distance_score
        * (0.35 + 0.25 * inverse_distance / (np.max(inverse_distance) + eps))
        * (0.55 + 0.25 * mutual_support + 0.20 * rank_support)
    )

    priors[~finite] = 0.0
    np.fill_diagonal(priors, 0.0)

    maximum = float(np.max(priors))
    if maximum > 0.0 and np.isfinite(maximum):
        priors /= maximum

    return priors
