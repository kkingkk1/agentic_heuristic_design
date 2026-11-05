import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric triangle-shortcut edge heuristic matrix."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    if np.any(np.isfinite(distances) & (distances < 0)):
        raise ValueError("distance_matrix must contain non-negative distances")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    positive = finite & (distances > 0)
    usable = distances[positive]

    if usable.size == 0:
        return scores

    scale = float(np.median(usable))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    symmetric = 0.5 * (distances + distances.T)
    valid_edges = np.isfinite(symmetric)
    np.fill_diagonal(valid_edges, False)

    safe = np.where(valid_edges, symmetric, np.inf)
    np.fill_diagonal(safe, np.inf)

    best_alternative = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        via_k = safe[:, k, None] + safe[None, k, :]
        via_k[k, :] = np.inf
        via_k[:, k] = np.inf
        best_alternative = np.minimum(best_alternative, via_k)

    direct = np.where(valid_edges, symmetric, 0.0)
    margin = (best_alternative - direct) / scale

    clipped_margin = np.clip(margin, -40.0, 40.0)
    exclusivity = 1.0 / (1.0 + np.exp(-clipped_margin))
    exclusivity[~valid_edges] = 0.0

    direct_preference = 1.0 / (1.0 + direct / scale)
    direct_preference[~valid_edges] = 0.0

    scores = direct_preference * exclusivity
    scores = 0.5 * (scores + scores.T)
    scores[~valid_edges] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum
    else:
        scores.fill(0.0)

    return scores
