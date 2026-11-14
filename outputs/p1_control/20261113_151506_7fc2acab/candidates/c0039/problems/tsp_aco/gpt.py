import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally balanced stochastic priors for TSP edges."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(distances < 0):
        raise ValueError("distance_matrix must be non-negative")

    n = distances.shape[0]
    result = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return result

    work = distances.copy()
    np.fill_diagonal(work, np.inf)

    finite_edges = work[np.isfinite(work)]
    if finite_edges.size == 0:
        return result

    center = float(np.median(finite_edges))
    deviation = float(np.median(np.abs(finite_edges - center)))
    temperature = max(deviation, np.finfo(float).eps)

    shifted = np.where(np.isfinite(work), work - np.min(finite_edges), 0.0)
    kernel = np.exp(-np.minimum(shifted / temperature, 60.0))
    kernel[~np.isfinite(work)] = 0.0
    np.fill_diagonal(kernel, 0.0)

    for _ in range(120):
        row_sum = kernel.sum(axis=1, keepdims=True)
        kernel = np.divide(
            kernel,
            row_sum,
            out=np.zeros_like(kernel),
            where=row_sum > 0,
        )

        col_sum = kernel.sum(axis=0, keepdims=True)
        kernel = np.divide(
            kernel,
            col_sum,
            out=np.zeros_like(kernel),
            where=col_sum > 0,
        )

    profile_similarity = kernel @ kernel.T
    profile_scale = float(np.max(profile_similarity))
    if profile_scale > 0.0 and np.isfinite(profile_scale):
        profile_similarity /= profile_scale

    score = kernel * (0.7 + 0.3 * profile_similarity)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
