import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute edge-priority indicators for a TSP distance matrix.

    Higher values indicate edges that are more promising for inclusion.
    The output has the same shape as distance_matrix.
    """
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty_like(distances, dtype=float)
    if n == 1:
        return np.zeros_like(distances, dtype=float)

    finite = np.isfinite(distances)
    positive = finite & (distances > 0)

    if not np.any(positive):
        return np.zeros_like(distances, dtype=float)

    valid_values = distances[positive]
    scale = np.median(valid_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.mean(valid_values)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    safe_distances = np.where(positive, distances, np.inf)

    # Base preference for short edges, using a robust exponential decay.
    inverse_score = np.zeros_like(distances, dtype=float)
    inverse_score[positive] = np.exp(-safe_distances[positive] / scale)

    # Rank-based preference: close edges receive stronger nonlinear emphasis.
    nearest = np.min(safe_distances, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, scale)
    relative = np.zeros_like(distances, dtype=float)
    relative[positive] = nearest[:, None][positive] / safe_distances[positive]
    relative = np.clip(relative, 0.0, 1.0)

    # Mutual-nearest-neighbor evidence.
    row_best = np.zeros_like(distances, dtype=float)
    for i in range(n):
        row = safe_distances[i].copy()
        row[i] = np.inf
        order = np.argsort(row)
        usable = order[np.isfinite(row[order])]
        if usable.size:
            row_best[i, usable[: min(3, usable.size)]] = 1.0

    mutual = row_best * row_best.T

    # Favor edges incident to nodes whose alternatives are comparatively long.
    local_mean = np.zeros(n, dtype=float)
    for i in range(n):
        row = safe_distances[i]
        finite_row = row[np.isfinite(row)]
        local_mean[i] = np.mean(finite_row) if finite_row.size else scale

    sparsity = np.zeros_like(distances, dtype=float)
    denominator = np.maximum(local_mean[:, None] + local_mean[None, :], 1e-12)
    sparsity[positive] = np.clip(
        2.0 * safe_distances[positive] / denominator[positive], 0.0, 2.0
    )
    sparsity = 1.0 - np.abs(sparsity - 1.0)

    score = (
        0.60 * inverse_score
        + 0.25 * relative
        + 0.10 * mutual
        + 0.05 * sparsity
    )

    # TSP edges are undirected; enforce symmetry even for asymmetric input.
    score = 0.5 * (score + score.T)
    score[~np.isfinite(score)] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0:
        score /= maximum

    return score.astype(float, copy=False)
