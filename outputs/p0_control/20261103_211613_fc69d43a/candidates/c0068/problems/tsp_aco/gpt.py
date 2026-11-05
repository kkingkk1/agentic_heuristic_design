import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-preference matrix for a minimization TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    d = np.maximum((distances + distances.T) * 0.5, 0.0)
    np.fill_diagonal(d, 0.0)

    positive = d[d > 0.0]
    distance_scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )
    distance_scale = max(distance_scale, np.finfo(float).eps)

    normalized = d / distance_scale

    visibility = 1.0 / np.maximum(normalized, 1e-12)
    np.fill_diagonal(visibility, 0.0)

    third_nodes = np.arange(n)
    replacement = (
        normalized[:, :, None] + normalized[None, :, :]
    )

    endpoint_mask = (
        (third_nodes[None, :, None] == third_nodes[:, None, None])
        | (third_nodes[None, :, None] == third_nodes[None, None, :])
    )
    replacement = np.where(endpoint_mask, np.inf, replacement)

    best_replacement = np.min(replacement, axis=1)
    regret = np.maximum(best_replacement - normalized, 0.0)
    np.fill_diagonal(regret, 0.0)

    positive_regret = regret[regret > 0.0]
    regret_scale = (
        float(np.median(positive_regret))
        if positive_regret.size
        else 1.0
    )
    regret_scale = max(regret_scale, np.finfo(float).eps)

    distinctiveness = np.minimum(regret / regret_scale, 8.0)
    scores = visibility * (1.0 + 0.75 * distinctiveness)

    scores = (scores + scores.T) * 0.5
    np.fill_diagonal(scores, 0.0)

    scores[~np.isfinite(scores)] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
