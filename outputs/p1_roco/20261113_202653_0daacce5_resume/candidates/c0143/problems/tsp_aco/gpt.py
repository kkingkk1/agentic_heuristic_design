import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return robust normalized edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    priors = np.zeros((n, n), dtype=float)

    if n == 0:
        return priors

    valid = np.isfinite(distances) & (distances > 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return priors

    positive = distances[valid]
    scale = float(np.median(positive)) if positive.size else 1.0
    eps = np.sqrt(np.finfo(float).eps) * max(scale, 1.0)

    inverse_distance = np.zeros_like(distances, dtype=float)
    inverse_distance[valid] = 1.0 / (distances[valid] + eps)

    inverse_values = inverse_distance[valid]
    cap = float(np.quantile(inverse_values, 0.99))
    cap = max(cap, eps)
    inverse_distance = np.minimum(inverse_distance, cap)

    base_scale = float(np.quantile(inverse_distance[valid], 0.99))
    if base_scale > 0.0:
        inverse_signal = inverse_distance / base_scale
    else:
        inverse_signal = np.zeros_like(inverse_distance)

    row_best = np.max(inverse_signal, axis=1, keepdims=True)
    col_best = np.max(inverse_signal, axis=0, keepdims=True)

    row_competitiveness = np.divide(
        inverse_signal,
        row_best,
        out=np.zeros_like(inverse_signal),
        where=row_best > 0.0,
    )
    col_competitiveness = np.divide(
        inverse_signal,
        col_best,
        out=np.zeros_like(inverse_signal),
        where=col_best > 0.0,
    )

    scores = (
        0.55 * inverse_signal
        + 0.25 * row_competitiveness
        + 0.20 * col_competitiveness
    )

    scores[~valid] = 0.0

    score_max = float(np.max(scores))
    if score_max <= 0.0:
        priors[valid] = 1.0
        return priors

    scores /= score_max

    exploration_floor = 0.02
    scores[valid] = exploration_floor + (1.0 - exploration_floor) * scores[valid]
    scores[~valid] = 0.0

    return scores
