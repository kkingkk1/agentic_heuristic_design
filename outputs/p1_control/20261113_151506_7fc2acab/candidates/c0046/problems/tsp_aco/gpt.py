import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric distance-based prior for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if np.any(np.isfinite(distances) & (distances < 0)):
        raise ValueError("distance_matrix must contain non-negative distances")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    valid = np.isfinite(distances) & (distances >= 0)
    np.fill_diagonal(valid, False)

    work = np.where(valid, distances, np.inf)

    finite_values = work[valid]
    positive_values = finite_values[finite_values > 0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.median(finite_values))
    else:
        global_scale = 1.0

    eps = np.finfo(float).eps
    global_scale = max(global_scale, eps)

    nearest = np.min(work, axis=1)
    nearest = np.where(
        np.isfinite(nearest) & (nearest > 0),
        nearest,
        global_scale,
    )

    positive_work = np.where(valid & (work > 0), work, np.inf)
    k = max(1, min(n - 1, int(np.sqrt(n))))
    local_scale = np.partition(positive_work, k - 1, axis=1)[:, :k]
    local_scale = np.where(np.isfinite(local_scale), local_scale, global_scale)
    local_scale = np.maximum(np.mean(local_scale, axis=1), eps)

    safe_distance = np.where(valid, work, 0.0)

    inverse_distance = np.zeros_like(work)
    inverse_distance[valid] = 1.0 / (
        safe_distance[valid] + global_scale * 1e-6
    )

    relative_shortness = np.zeros_like(work)
    relative_shortness[valid] = np.exp(
        -safe_distance[valid]
        / (np.sqrt(local_scale[:, None] * local_scale[None, :])[valid] + eps)
    )

    ranks = np.full((n, n), n, dtype=float)
    for i in range(n):
        candidates = np.flatnonzero(valid[i])
        if candidates.size:
            order = candidates[np.argsort(work[i, candidates], kind="stable")]
            ranks[i, order] = np.arange(1, order.size + 1, dtype=float)

    rank_scale = max(1.0, np.sqrt(n))
    endpoint_preference = np.zeros_like(work)
    endpoint_preference[valid] = np.exp(
        -0.5
        * (
            ranks[valid] / rank_scale
            + ranks.T[valid] / rank_scale
        )
    )

    nearest_indices = np.argmin(work, axis=1)
    mutual_bonus = np.zeros_like(work)
    for i in range(n):
        j = nearest_indices[i]
        if valid[i, j] and valid[j, i] and nearest_indices[j] == i:
            mutual_bonus[i, j] = 1.0
            mutual_bonus[j, i] = 1.0

    score = (
        inverse_distance
        * (0.55 + 0.45 * relative_shortness)
        * (0.60 + 0.40 * endpoint_preference)
        * (1.0 + 0.20 * mutual_bonus)
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score)) if score.size else 0.0
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
