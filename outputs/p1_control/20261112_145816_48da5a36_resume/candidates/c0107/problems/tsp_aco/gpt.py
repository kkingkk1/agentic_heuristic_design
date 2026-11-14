import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-promise indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    distances = 0.5 * (distances + distances.T)
    diagonal = np.eye(n, dtype=bool)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0) & ~diagonal

    if not np.any(valid):
        return result

    positive = distances[valid & (distances > 0.0)]
    eps = np.finfo(float).eps
    global_scale = float(np.median(positive)) if positive.size else 1.0
    global_scale = max(global_scale, eps)

    fallback = float(np.max(positive)) if positive.size else global_scale
    fallback = max(fallback * 10.0, global_scale)

    safe = np.where(valid, distances, fallback)
    np.fill_diagonal(safe, fallback)

    positive_safe = np.where(
        valid & (safe > 0.0),
        safe,
        np.inf,
    )
    sorted_positive = np.sort(positive_safe, axis=1)

    nearest = sorted_positive[:, 0]
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0.0),
        nearest,
        global_scale,
    )

    k = min(max(2, int(np.sqrt(n))), n - 1)
    neighborhood = sorted_positive[:, :k]
    local_scale = np.mean(
        np.where(np.isfinite(neighborhood), neighborhood, global_scale),
        axis=1,
    )
    local_scale = np.maximum(local_scale, eps)

    endpoint_scale = np.sqrt(
        np.outer(local_scale, local_scale)
    )

    relative_distance = safe / np.maximum(endpoint_scale, eps)
    proximity = np.exp(-np.clip(relative_distance, 0.0, 50.0))

    order = np.argsort(safe, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n, dtype=float)[None, :]
    rank_score = 1.0 / (1.0 + ranks)
    rank_score[~valid] = 0.0

    mutual_rank = np.sqrt(rank_score * rank_score.T)
    mutual_rank /= max(float(np.max(mutual_rank)), eps)

    density = global_scale / np.maximum(local_scale, eps)
    density_ratio = np.abs(
        np.log(
            np.maximum(density[:, None], eps)
            / np.maximum(density[None, :], eps)
        )
    )
    density_contrast = density_ratio / (1.0 + density_ratio)

    nearest_indices = np.argsort(safe, axis=1, kind="stable")[:, :k]
    neighbor_mask = np.zeros((n, n), dtype=bool)
    neighbor_mask[rows, nearest_indices] = True
    neighbor_mask &= ~diagonal

    overlap = neighbor_mask.astype(float) @ neighbor_mask.astype(float).T
    overlap /= float(max(k, 1))
    overlap = np.clip(overlap, 0.0, 1.0)

    inverse_distance = global_scale / np.maximum(safe, eps)
    inverse_distance = np.clip(inverse_distance, 0.0, 1.0)

    bridge_factor = 0.7 + 0.3 * density_contrast
    cluster_factor = 1.0 - 0.15 * overlap

    log_score = (
        0.85 * np.log(np.maximum(inverse_distance, eps))
        + 0.75 * np.log(np.maximum(proximity, eps))
        + 1.10 * np.log(np.maximum(0.25 + 0.75 * mutual_rank, eps))
        + np.log(np.maximum(bridge_factor, eps))
        + np.log(np.maximum(cluster_factor, eps))
    )

    zero_edges = valid & (distances == 0.0)
    log_score[zero_edges] += 2.0
    log_score[~valid] = -np.inf

    finite_scores = np.isfinite(log_score)
    if not np.any(finite_scores):
        return result

    maximum = float(np.max(log_score[finite_scores]))
    result = np.exp(np.where(finite_scores, log_score - maximum, -np.inf))
    result[~valid] = 0.0
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)

    maximum = float(np.max(result))
    if maximum > 0.0:
        result /= maximum

    return result.astype(float, copy=False)
