import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return symmetric edge-priority indicators for stochastic TSP sampling.

    Larger values indicate edges that are both short and selectively preferred
    by their endpoints. Self-loops and invalid edges receive zero.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    finite_off_diagonal = np.isfinite(distances)
    finite_off_diagonal &= ~np.eye(n, dtype=bool)

    positive = finite_off_diagonal & (distances > 0)
    if not np.any(positive):
        return np.zeros((n, n), dtype=float)

    positive_values = distances[positive]
    fallback = float(np.median(positive_values))

    safe = distances.copy()
    safe[~np.isfinite(safe)] = fallback
    safe = np.maximum(safe, np.finfo(float).eps)
    np.fill_diagonal(safe, np.inf)

    valid = np.isfinite(safe) & ~np.eye(n, dtype=bool)

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / safe[valid]

    row_best = np.max(inverse, axis=1, keepdims=True)
    row_best[row_best <= 0] = 1.0
    row_affinity = inverse / row_best

    # Soft local competition: an edge is more selective when few alternatives
    # have comparable row-normalized affinity.
    temperature = 5.0
    alternatives = np.where(valid, row_affinity, 0.0)
    soft_mass = np.sum(np.exp(temperature * alternatives), axis=1) - 1.0
    soft_mass = np.maximum(soft_mass, 1e-12)

    endpoint_selectivity = np.zeros(n, dtype=float)
    endpoint_selectivity[soft_mass > 0] = 1.0 / soft_mass[soft_mass > 0]

    if np.max(endpoint_selectivity) > 0:
        endpoint_selectivity /= np.max(endpoint_selectivity)

    scarcity = np.sqrt(
        endpoint_selectivity[:, None] * endpoint_selectivity[None, :]
    )

    directed_score = row_affinity * scarcity
    score = 0.5 * (directed_score + directed_score.T)

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
