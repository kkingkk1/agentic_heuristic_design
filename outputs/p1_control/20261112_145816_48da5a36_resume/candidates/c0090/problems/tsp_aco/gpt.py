import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(distances)
    finite_values = distances[finite]
    nonnegative = finite_values[finite_values >= 0]

    replacement = float(np.max(nonnegative)) if nonnegative.size else 1.0
    distances = np.where(finite, distances, replacement)
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    np.fill_diagonal(distances, np.inf)

    positive = distances[np.isfinite(distances) & (distances > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    eps = np.finfo(float).eps

    safe_distances = np.maximum(distances, eps)

    inverse_distance = 1.0 / safe_distances
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    inverse_max = float(np.max(inverse_distance))
    if inverse_max > 0:
        inverse_distance /= inverse_max

    row_sorted = np.sort(safe_distances, axis=1)
    local_scale = row_sorted[:, 0]
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        fallback,
    )
    local_scale = np.maximum(local_scale, eps)

    relative_proximity = np.exp(
        -distances / np.sqrt(np.outer(local_scale, local_scale))
    )
    relative_proximity[~np.isfinite(relative_proximity)] = 0.0
    np.fill_diagonal(relative_proximity, 0.0)

    ranks = np.argsort(np.argsort(safe_distances, axis=1), axis=1)
    rank_score = 1.0 / (1.0 + ranks.astype(float))
    rank_score /= np.maximum(
        np.max(rank_score, axis=1, keepdims=True),
        eps,
    )
    mutual_preference = np.sqrt(rank_score * rank_score.T)
    np.fill_diagonal(mutual_preference, 0.0)

    base_score = (
        0.50 * inverse_distance
        + 0.30 * relative_proximity
        + 0.20 * mutual_preference
    )

    k = min(max(2, int(np.sqrt(n))), n - 1)
    neighbor_mask = np.zeros((n, n), dtype=bool)
    nearest_indices = np.argsort(safe_distances, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    neighbor_mask[rows, nearest_indices] = True
    np.fill_diagonal(neighbor_mask, False)

    overlap = neighbor_mask.astype(float) @ neighbor_mask.astype(float).T
    overlap /= float(max(k, 1))
    overlap = np.clip(overlap, 0.0, 1.0)

    density = 1.0 / local_scale
    density /= max(float(np.median(density)), eps)
    density_ratio = np.abs(
        np.log(np.maximum(density[:, None], eps)
               / np.maximum(density[None, :], eps))
    )
    density_contrast = density_ratio / (1.0 + density_ratio)

    bridge_signal = relative_proximity * (0.55 + 0.45 * density_contrast)
    bridge_signal = 0.5 * (bridge_signal + bridge_signal.T)

    cluster_penalty = 1.0 - 0.12 * overlap
    score = (
        0.84 * base_score
        + 0.16 * bridge_signal
    ) * cluster_penalty

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
