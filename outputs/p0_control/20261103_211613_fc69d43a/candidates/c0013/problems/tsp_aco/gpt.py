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
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    normalized = d / scale

    visibility = 1.0 / (normalized + 1e-12)
    np.fill_diagonal(visibility, 0.0)

    replacement = normalized[:, :, None] + normalized[None, :, :]
    replacement[:, np.arange(n), np.arange(n)] = np.inf
    replacement[np.arange(n), :, :] = np.where(
        np.eye(n, dtype=bool)[:, :, None],
        np.inf,
        replacement[np.arange(n), :, :],
    )

    best_replacement = np.min(replacement, axis=1)
    regret = np.maximum(best_replacement - normalized, 0.0)

    positive_regret = regret[regret > 0.0]
    regret_scale = (
        float(np.median(positive_regret))
        if positive_regret.size
        else 1.0
    )
    regret_scale = max(regret_scale, np.finfo(float).eps)

    distinctiveness = regret / regret_scale
    scores = visibility * (1.0 + distinctiveness)

    scores = (scores + scores.T) * 0.5
    np.fill_diagonal(scores, 0.0)

    finite_scores = np.isfinite(scores)
    if not np.all(finite_scores):
        scores[~finite_scores] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
