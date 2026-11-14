import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps

    finite = np.isfinite(distances)
    finite_values = distances[finite]
    nonnegative = finite_values[finite_values >= 0.0]
    replacement = float(np.max(nonnegative)) if nonnegative.size else 1.0

    distances = np.where(finite, distances, replacement)
    distances = np.maximum(distances, 0.0)
    distances = 0.5 * (distances + distances.T)

    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    np.fill_diagonal(distances, np.inf)

    positive_mask = np.isfinite(distances) & (distances > 0.0)
    positive_distances = np.where(positive_mask, distances, np.inf)
    local_scale = np.min(positive_distances, axis=1)
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0.0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    safe_distances = np.where(np.isfinite(distances), distances, global_scale)

    inverse_distance = 1.0 / np.maximum(safe_distances, eps)
    inverse_distance[~np.isfinite(inverse_distance)] = 0.0
    inverse_max = float(np.max(inverse_distance))
    if inverse_max > 0.0:
        inverse_distance /= inverse_max

    normalized_length = safe_distances / np.sqrt(
        local_scale[:, None] * local_scale[None, :]
    )
    proximity = np.exp(-np.minimum(normalized_length, 50.0))
    proximity[~np.isfinite(proximity)] = 0.0

    order = np.argsort(safe_distances, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)

    reciprocal_rank = 1.0 / (1.0 + ranks)
    reciprocal_rank /= np.maximum(
        np.max(reciprocal_rank, axis=1, keepdims=True),
        eps,
    )
    mutual_rank = np.sqrt(reciprocal_rank * reciprocal_rank.T)

    k = min(max(2, int(np.sqrt(n))), n - 1)
    nearest = order[:, :k]
    neighbor_mask = np.zeros((n, n), dtype=float)
    neighbor_mask[rows, nearest] = 1.0
    np.fill_diagonal(neighbor_mask, 0.0)

    overlap = neighbor_mask @ neighbor_mask.T
    overlap = np.clip(overlap / float(max(k, 1)), 0.0, 1.0)

    log_scale_difference = np.abs(
        np.log(local_scale[:, None]) - np.log(local_scale[None, :])
    )
    density_contrast = log_scale_difference / (1.0 + log_scale_difference)

    endpoint_proximity = np.sqrt(proximity * proximity.T)

    log_core = (
        0.48 * np.log(np.maximum(inverse_distance, eps))
        + 0.32 * np.log(np.maximum(endpoint_proximity, eps))
        + 0.20 * np.log(np.maximum(mutual_rank, eps))
    )
    log_core -= np.max(log_core)

    core = np.exp(np.clip(log_core, -745.0, 0.0))

    bridge_adjustment = 0.88 + 0.12 * density_contrast
    cluster_adjustment = 1.0 - 0.10 * overlap
    score = core * bridge_adjustment * cluster_adjustment

    zero_edges = distances == 0.0
    score[zero_edges] = np.maximum(score[zero_edges], 1.0)

    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
