import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-preference matrix for a minimization TSP solver."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    d = 0.5 * (distances + distances.T)
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, 0.0)

    eps = np.finfo(float).eps
    positive_distances = d[d > 0.0]
    distance_scale = (
        float(np.median(positive_distances))
        if positive_distances.size
        else 1.0
    )
    distance_scale = max(distance_scale, eps)

    normalized = d / distance_scale

    with np.errstate(divide="ignore", invalid="ignore"):
        visibility = 1.0 / np.maximum(normalized, eps)
    np.fill_diagonal(visibility, 0.0)

    if n >= 3:
        indices = np.arange(n)

        # detours[i, j, k] = d[i, k] + d[k, j]
        detours = normalized[:, None, :] + normalized[None, :, :]

        valid_third_node = (
            (indices[None, None, :] != indices[:, None, None])
            & (indices[None, None, :] != indices[None, :, None])
        )
        detours = np.where(valid_third_node, detours, np.inf)

        best_detour = np.min(detours, axis=2)
        regret = np.maximum(best_detour - normalized, 0.0)
        regret[~np.isfinite(regret)] = 0.0
    else:
        regret = np.zeros_like(normalized)

    positive_regret = regret[regret > 0.0]
    regret_scale = (
        float(np.median(positive_regret))
        if positive_regret.size
        else 1.0
    )
    regret_scale = max(regret_scale, eps)

    distinctiveness = regret / regret_scale
    scores = visibility * (1.0 + distinctiveness)

    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores[~np.isfinite(scores)] = 0.0

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores
